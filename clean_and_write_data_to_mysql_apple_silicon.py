from transformers import pipeline, logging, AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import torch, json
from torch.quantization import quantize_dynamic
from datetime import datetime
from sqlalchemy import create_engine
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tqdm import tqdm
from sqlalchemy.types import JSON
from pathlib import Path

# ── Device detection ──────────────────────────────────────────────────────────
def get_device() -> str:
    """
    Returns the best available device string for PyTorch.
    Prefers Apple MPS, then CUDA, then CPU.
    """
    if torch.backends.mps.is_available():
        print("✅ Apple Silicon MPS detected — using MPS")
        return "mps"
    elif torch.cuda.is_available():
        print("✅ CUDA GPU detected — using CUDA")
        return "cuda"
    else:
        print("⚠️  No GPU detected — falling back to CPU")
        return "cpu"

DEVICE = get_device()

# ── Model map ─────────────────────────────────────────────────────────────────
MODEL_MAP = {
    "sentiment": "distilbert-base-uncased-finetuned-sst-2-english",
    "emotion":   "j-hartmann/emotion-english-distilroberta-base"
}

# ── Utility functions ─────────────────────────────────────────────────────────
def chunk_dataframe(df, chunk_size=None):
    """
    Splits a DataFrame into sequential chunks of a specified size, yielding each with its index.

    Parameters
    ----------
    df : pandas.DataFrame
    chunk_size : int or None — defaults to 10,000

    Yields
    ------
    Tuple[int, pandas.DataFrame]
    """
    if chunk_size is None:
        chunk_size = 10_000
    for c in range(0, len(df), chunk_size):
        yield c // chunk_size, df.iloc[c:c + chunk_size]


def merge_sentiment_chunks(save_dir, analysis_type):
    """Merge all chunk parquet files for a given analysis type."""
    files = sorted(save_dir.glob(f"{analysis_type}_chunk_*.parquet"))
    dfs = [pd.read_parquet(f) for f in files]
    return pd.concat(dfs, ignore_index=True)


def batched_scores(texts, analysis_pipeline, batch_size=16):
    """Run a HuggingFace pipeline in batches, returning a list of score dicts."""
    all_scores = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        with torch.no_grad():
            batch_outputs = analysis_pipeline(batch)
        batch_scores = [
            {item['label']: item['score'] for item in output}
            for output in batch_outputs
        ]
        all_scores.extend(batch_scores)
    return all_scores


def process_saved_chunks(analysis_type, df_name):
    """Load and return all saved chunk parquet files for a given analysis/df name."""
    files = list(Path(f"{analysis_type}_chunks").glob(f"{df_name}_{analysis_type}_chunk_*.parquet"))
    print(f"Looking for files in:  {analysis_type}_chunks")
    print(f"Pattern:               {df_name}_{analysis_type}_chunk_*.parquet")
    print(f"Matched files:         {[f.name for f in files]}")
    file_chunks = []
    for f in sorted(files):
        try:
            file_chunks.append(pd.read_parquet(f))
        except Exception as e:
            print(f"⚠️  Skipped {f.name}: {e}")
    return file_chunks


# ── Main class ────────────────────────────────────────────────────────────────
class ReviewAnalyzer:
    def __init__(
        self,
        analysis_type,
        model_name=None,
        data_frame=None,
        save_dir=None,
        chunk_size=1000,
        batch_size=16,
        text_column="review_text",
        df_name="reviews",
        use_quantized=False,
        device=None           # NEW: injectable device, defaults to auto-detected DEVICE
    ):
        self.analysis_type = analysis_type
        self.model_name    = model_name or "distilbert-base-uncased-finetuned-sst-2-english"
        self.data_frame    = data_frame
        self.chunk_size    = chunk_size
        self.batch_size    = batch_size
        self.text_column   = text_column
        self.df_name       = df_name
        self.save_dir      = Path(save_dir or f"{analysis_type}_chunks")
        self.save_dir.mkdir(exist_ok=True)

        # ── Device resolution ─────────────────────────────────────────────────
        # Priority: explicit arg → use_quantized forces CPU → global DEVICE
        if use_quantized:
            self.device = "cpu"
        else:
            self.device = device or DEVICE

        print(f"ReviewAnalyzer [{analysis_type}] using device: {self.device}")

        # ── Load model / pipeline ─────────────────────────────────────────────
        base_model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

        if analysis_type == "sentiment":
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if use_quantized:
                # Quantized models stay on CPU — dynamic quantisation not supported on MPS
                self.model = quantize_dynamic(base_model, {torch.nn.Linear}, dtype=torch.qint8)
            else:
                self.model = base_model.to(self.device)
            self.pipeline = None

        else:
            # For emotion (and any pipeline-based analysis):
            # HuggingFace pipeline accepts device as a string ("mps", "cpu")
            # or an integer CUDA index.  Use string form for MPS/CPU compatibility.
            pipeline_device = self.device if self.device != "cuda" else 0
            self.pipeline  = pipeline(
                "text-classification",
                model=self.model_name,
                device=pipeline_device,
                top_k=1
            )
            self.tokenizer = None
            self.model     = None

    # ── Checkpoint helpers ────────────────────────────────────────────────────
    def get_completed_chunks(self):
        completed = set()
        for f in self.save_dir.glob(f"{self.df_name}_{self.analysis_type}_chunk_*.parquet"):
            parts = f.stem.split("_")
            if self.analysis_type == "emotion":
                try:
                    start_val = int(parts[-2])
                    completed.add(start_val // self.chunk_size)
                except (ValueError, IndexError):
                    pass
            else:
                try:
                    completed.add(int(parts[-1]))
                except (ValueError, IndexError):
                    pass
        return completed

    def chunk_dataframe(self):
        for start in range(0, len(self.data_frame), self.chunk_size):
            yield start, self.data_frame.iloc[start:start + self.chunk_size]

    # ── Main entry point ──────────────────────────────────────────────────────
    def run(self):
        completed = self.get_completed_chunks()
        for start, chunk_df in tqdm(self.chunk_dataframe(), desc=f"Running {self.analysis_type} analysis"):
            chunk_id = start // self.chunk_size
            if chunk_id in completed:
                continue
            print(f"[{datetime.now()}] Processing chunk {chunk_id}...")

            if self.analysis_type == "sentiment":
                self._process_sentiment_chunk(chunk_id, chunk_df, "work_id", self.device)
            else:
                self._process_emotion_chunk(start, chunk_df)

    # ── Sentiment chunk ───────────────────────────────────────────────────────
    def _process_sentiment_chunk(self, chunk_id, chunk_df, key_column, device):
        texts    = chunk_df[self.text_column].tolist()
        all_preds = []

        for i in range(0, len(texts), self.batch_size):
            batch  = texts[i:i + self.batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )

            # Move inputs to the correct device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # autocast: supported on cuda and cpu, NOT on mps
            # On MPS we skip autocast — inference still runs fine
            if device == "cuda":
                ctx = torch.autocast("cuda")
            elif device == "cpu":
                ctx = torch.autocast("cpu")
            else:
                # MPS — use a no-op context
                import contextlib
                ctx = contextlib.nullcontext()

            with ctx, torch.no_grad():
                outputs = self.model(**inputs)
                preds   = torch.argmax(outputs.logits, dim=1)
                scores  = torch.softmax(outputs.logits, dim=1)

            for pred, score in zip(preds, scores):
                label      = "POSITIVE" if pred.item() == 1 else "NEGATIVE"
                confidence = score[pred].item()
                sentiment  = confidence if label == "POSITIVE" else -confidence
                all_preds.append({
                    "label_hf":    label,
                    "score_hf":    confidence,
                    "sentiment_hf": sentiment
                })

        result_df              = pd.DataFrame(all_preds)
        #result_df[key_column]  = chunk_df[key_column].values #Commented out to save review_id as well
        result_df['work_id'] = chunk_df['work_id'].values
        result_df['review_id'] = chunk_df['review_id'].values
        result_df.to_parquet(
            self.save_dir / f"{self.df_name}_{self.analysis_type}_chunk_{chunk_id}.parquet"
        )

    # ── Emotion chunk ─────────────────────────────────────────────────────────
    def _process_emotion_chunk(self, start, chunk_df):
        texts      = chunk_df[self.text_column].tolist()
        all_scores = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            with torch.no_grad():
                batch_outputs = self.pipeline(batch)
            batch_scores = [
                {item['label']: item['score'] for item in output}
                for output in batch_outputs
            ]
            all_scores.extend(batch_scores)

        chunk_df = chunk_df.copy()
        chunk_df["emotion_scores"] = all_scores
        end         = start + len(chunk_df)
        output_path = self.save_dir / f"{self.df_name}_{self.analysis_type}_chunk_{start}_{end}.parquet"
        chunk_df.to_parquet(output_path, index=False)
        print(f"[{datetime.now()}] ✅ Saved: {output_path}")

    # ── Accessors ─────────────────────────────────────────────────────────────
    def get_analysis_type(self): return self.analysis_type
    def get_data_frame(self):    return self.data_frame
    def get_df_name(self):       return self.df_name


# ═══════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════════════════

# ── Load works ────────────────────────────────────────────────────────────────
# Update path for your Mac — OneDrive path replaced with a generic placeholder
works = pd.read_csv("./Data/goodreads_works.csv")

works['original_publication_year'] = (
    pd.to_numeric(works['original_publication_year'], errors='coerce').fillna(0).astype(int)
)
works['num_pages'] = (
    pd.to_numeric(works['num_pages'], errors='coerce').fillna(0).astype(int)
)
works['isbn13']      = works['isbn13'].astype('string').str.replace(".0", "", regex=False)
works['description'] = works['description'].fillna('')

# ── Works: HuggingFace Sentiment ──────────────────────────────────────────────
logging.set_verbosity_error()

analyzer = ReviewAnalyzer(
    analysis_type="sentiment",
    model_name=MODEL_MAP["sentiment"],
    data_frame=works,
    save_dir="./sentiment_chunks",
    chunk_size=1000,
    batch_size=12,
    text_column="description",
    df_name="works",
    use_quantized=False
)
analyzer.run()

# ── Works: VADER Sentiment ────────────────────────────────────────────────────
vader_analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):
    return vader_analyzer.polarity_scores(str(text))['compound']

works['sentiment'] = works['description'].apply(get_sentiment)
print('Works: Applied VADER Sentiment Analysis - Time:', datetime.now())

# ── Merge HF sentiment chunks back onto works ─────────────────────────────────
chunks = process_saved_chunks(analyzer.get_analysis_type(), analyzer.get_df_name())
works_sentiments = pd.concat(chunks, ignore_index=True)

works_final = works.merge(
    works_sentiments[['work_id', 'label_hf', 'score_hf', 'sentiment_hf']],
    on='work_id',
    how='left'
)

# ── Works: Emotion Analysis ───────────────────────────────────────────────────
analyzer = ReviewAnalyzer(
    analysis_type="emotion",
    model_name=MODEL_MAP["emotion"],
    data_frame=works,
    save_dir="./emotion_chunks",
    chunk_size=1000,
    batch_size=12,
    text_column="description",
    df_name="works",
    use_quantized=False
)
analyzer.run()

chunks = process_saved_chunks(analyzer.get_analysis_type(), analyzer.get_df_name())
works_emotions = pd.concat(chunks, ignore_index=True)

works_emotions_final = works.merge(
    works_emotions[['work_id', 'emotion_scores']],
    on='work_id',
    how='left'
)
works_emotions_final['emotion_scores'] = works_emotions_final['emotion_scores'].apply(json.dumps)

# ── Write works to MySQL ──────────────────────────────────────────────────────
# Update connection string for your MySQL server
#engine = create_engine('mysql+pymysql://andrew:<password>@192.168.1.197:3306/mavenbookshelf')
engine = create_engine('mysql+pymysql://andrew:$H0nggh0rzaq!@localhost:3306/mavenbookshelf')
works_final.to_sql(
    'works',
    con=engine,
    if_exists='replace',
    index=False,
    dtype={'emotion_scores': JSON}
)
print('Works: Wrote to MySQL Server - Time:', datetime.now())

# ═══════════════════════════════════════════════════════════════════════════════
# Reviews pipeline
# ═══════════════════════════════════════════════════════════════════════════════

# ── Load reviews ──────────────────────────────────────────────────────────────
print('Starting Reviews - Time:', datetime.now())
reviews = pd.read_csv("./Data/goodreads_reviews.csv", low_memory=False)

# ── Reviews: HuggingFace Sentiment ────────────────────────────────────────────
analyzer = ReviewAnalyzer(
    analysis_type="sentiment",
    model_name=MODEL_MAP["sentiment"],
    data_frame=reviews,
    save_dir="./sentiment_chunks",
    chunk_size=10000,
    batch_size=16,
    text_column="review_text",
    df_name="reviews",
    use_quantized=False
)
analyzer.run()

# ── Reviews: VADER Sentiment ──────────────────────────────────────────────────
reviews['sentiment'] = reviews['review_text'].apply(get_sentiment)
print('Reviews: Applied VADER Sentiment Analysis - Time:', datetime.now())

# ── Merge HF sentiment chunks back onto reviews ───────────────────────────────
chunks = process_saved_chunks(analyzer.get_analysis_type(), analyzer.get_df_name())
reviews_sentiments = pd.concat(chunks, ignore_index=True)
reviews_sentiments = reviews_sentiments.drop_duplicates(subset='work_id')

reviews_final = reviews.merge(
    reviews_sentiments[['work_id', 'label_hf', 'score_hf', 'sentiment_hf']],
    on='work_id',
    how='left'
)

# ── Reviews: Emotion Analysis ─────────────────────────────────────────────────
analyzer = ReviewAnalyzer(
    analysis_type="emotion",
    model_name=MODEL_MAP["emotion"],
    data_frame=reviews,
    save_dir="./emotion_chunks",
    chunk_size=10000,
    batch_size=24,
    text_column="review_text",
    df_name="reviews",
    use_quantized=False
)
print('Reviews: Starting Emotion Analysis - Time:', datetime.now())
analyzer.run()

chunks = process_saved_chunks(analyzer.get_analysis_type(), analyzer.get_df_name())
reviews_emotions = pd.concat(chunks, ignore_index=True)
reviews_emotions = reviews_emotions.drop_duplicates(subset='work_id')

print("reviews:", reviews.shape)
print("reviews_emotions:", reviews_emotions.shape)

# ── Merge sentiment and emotion onto reviews ──────────────────────────────────
reviews_combined = reviews.merge(
    reviews_sentiments[['work_id', 'label_hf', 'score_hf', 'sentiment_hf']],
    on='work_id',
    how='left'
)
print("After sentiment merge:", reviews_combined.shape)

reviews_combined = reviews_combined.merge(
    reviews_emotions[['work_id', 'emotion_scores']],
    on='work_id',
    how='left'
)
print("After emotion merge:", reviews_combined.shape)

reviews_combined['emotion_scores'] = reviews_combined['emotion_scores'].apply(json.dumps)

# ── Write reviews to MySQL ────────────────────────────────────────────────────
reviews_combined.to_sql(
    'reviews',
    con=engine,
    if_exists='replace',
    index=False,
    dtype={'emotion_scores': JSON}
)
print('Reviews: Wrote to MySQL Database - Time:', datetime.now())