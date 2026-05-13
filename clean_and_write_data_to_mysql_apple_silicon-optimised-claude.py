from transformers import pipeline, logging, AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import torch, json, gc, contextlib
from torch.quantization import quantize_dynamic
from datetime import datetime
from sqlalchemy import create_engine
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tqdm import tqdm
from sqlalchemy.types import JSON
from pathlib import Path


# ── Device detection ──────────────────────────────────────────────────────────
def get_device() -> str:
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

MODEL_MAP = {
    "sentiment": "distilbert-base-uncased-finetuned-sst-2-english",
    "emotion":   "j-hartmann/emotion-english-distilroberta-base"
}

vader_analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):
    return vader_analyzer.polarity_scores(str(text))['compound']


# ── Streaming DB writer ───────────────────────────────────────────────────────
def write_to_db_streaming(df_name, engine, table_name, chunk_size,
                          sentiment_dir="sentiment_chunks",
                          emotion_dir="emotion_chunks"):
    """
    Write a dataset to the DB by streaming through saved emotion chunk files
    one at a time. Each emotion chunk already contains the full row data plus
    VADER sentiment (because VADER runs before emotion analysis). We merge HF
    sentiment from the matching sentiment chunk, then write.

    Peak memory per iteration: ~one emotion chunk + one sentiment chunk.
    This avoids holding the entire merged dataset (e.g. 1.15 M reviews) in RAM.

    IMPORTANT: chunk_size must match the value used when running sentiment
    analysis so that chunk IDs line up between the two sets of files.

    Join key: reviews are merged on review_id (unique per row) to avoid
    many-to-many blowup caused by merging on work_id (shared across reviews).
    Works are merged on work_id as before.
    """
    sentiment_path = Path(sentiment_dir)
    emotion_path   = Path(emotion_dir)

    # reviews have a unique review_id; works use work_id
    join_key = "review_id" if df_name == "reviews" else "work_id"

    emotion_files = sorted(emotion_path.glob(f"{df_name}_emotion_chunk_*.parquet"))
    if not emotion_files:
        print(f"⚠️  No emotion chunks found for {df_name} in {emotion_dir}")
        return

    for i, ef in enumerate(tqdm(emotion_files, desc=f"Writing {df_name} → {table_name}")):
        # Filename: {df_name}_emotion_chunk_{start}_{end}.parquet
        parts    = ef.stem.split("_")
        start    = int(parts[-2])
        chunk_id = start // chunk_size

        emot_df = pd.read_parquet(ef)

        sf = sentiment_path / f"{df_name}_sentiment_chunk_{chunk_id}.parquet"
        if sf.exists():
            sent_df = pd.read_parquet(sf)
            sent_df = sent_df.drop_duplicates(subset=join_key)
            emot_df = emot_df.merge(
                sent_df[[join_key, 'label_hf', 'score_hf', 'sentiment_hf']],
                on=join_key,
                how='left'
            )
            del sent_df
        else:
            print(f"⚠️  Sentiment chunk missing for chunk_id={chunk_id}, skipping HF merge")

        emot_df['emotion_scores'] = emot_df['emotion_scores'].apply(json.dumps)

        # First chunk uses 'replace' to drop/recreate the table with the correct schema
        # (including emotion_scores). Subsequent chunks append to that fresh table.
        if_exists = 'replace' if i == 0 else 'append'

        emot_df.to_sql(
            table_name,
            con=engine,
            if_exists=if_exists,
            index=False,
            dtype={'emotion_scores': JSON},
            chunksize=500,
        )
        del emot_df
        gc.collect()

    print(f"✅ {df_name} → {table_name} complete")


# ── Main analysis class ───────────────────────────────────────────────────────
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
        device=None,
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

        if use_quantized:
            self.device = "cpu"
        else:
            self.device = device or DEVICE

        print(f"ReviewAnalyzer [{analysis_type}] using device: {self.device}")

        base_model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

        if analysis_type == "sentiment":
            self.tokenizer   = AutoTokenizer.from_pretrained(self.model_name)
            self.model       = (
                quantize_dynamic(base_model, {torch.nn.Linear}, dtype=torch.qint8)
                if use_quantized
                else base_model.to(self.device)
            )
            self.hf_pipeline = None

        else:
            pipeline_device  = self.device if self.device != "cuda" else 0
            self.hf_pipeline = pipeline(
                "text-classification",
                model=self.model_name,
                device=pipeline_device,
                top_k=1,
            )
            self.tokenizer = None
            self.model     = None

    def cleanup(self):
        """Release model weights and pipeline from memory."""
        self.model       = None
        self.hf_pipeline = None
        self.tokenizer   = None
        self.data_frame  = None
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"✅ [{self.analysis_type}] model released from memory")

    # ── Checkpoint helpers ────────────────────────────────────────────────────
    def get_completed_chunks(self):
        completed = set()
        for f in self.save_dir.glob(f"{self.df_name}_{self.analysis_type}_chunk_*.parquet"):
            parts = f.stem.split("_")
            try:
                if self.analysis_type == "emotion":
                    # filename: …_chunk_{start}_{end}.parquet → start is second-to-last part
                    completed.add(int(parts[-2]) // self.chunk_size)
                else:
                    # filename: …_chunk_{chunk_id}.parquet
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
                self._process_sentiment_chunk(chunk_id, chunk_df, self.device)
            else:
                self._process_emotion_chunk(start, chunk_df)

    # ── Sentiment chunk ───────────────────────────────────────────────────────
    def _process_sentiment_chunk(self, chunk_id, chunk_df, device):
        texts     = chunk_df[self.text_column].fillna('').tolist()
        all_preds = []

        for i in range(0, len(texts), self.batch_size):
            batch  = texts[i:i + self.batch_size]
            inputs = self.tokenizer(
                batch, padding=True, truncation=True, max_length=512, return_tensors="pt"
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            if device == "cuda":
                ctx = torch.autocast("cuda")
            elif device == "cpu":
                ctx = torch.autocast("cpu")
            else:
                ctx = contextlib.nullcontext()   # MPS: autocast not supported

            with ctx, torch.no_grad():
                outputs = self.model(**inputs)
                preds   = torch.argmax(outputs.logits, dim=1)
                scores  = torch.softmax(outputs.logits, dim=1)

            for pred, score in zip(preds, scores):
                label      = "POSITIVE" if pred.item() == 1 else "NEGATIVE"
                confidence = score[pred].item()
                all_preds.append({
                    "label_hf":     label,
                    "score_hf":     confidence,
                    "sentiment_hf": confidence if label == "POSITIVE" else -confidence,
                })

        result_df            = pd.DataFrame(all_preds)
        result_df['work_id'] = chunk_df['work_id'].values
        if 'review_id' in chunk_df.columns:
            result_df['review_id'] = chunk_df['review_id'].values

        result_df.to_parquet(
            self.save_dir / f"{self.df_name}_{self.analysis_type}_chunk_{chunk_id}.parquet"
        )

    # ── Emotion chunk ─────────────────────────────────────────────────────────
    def _process_emotion_chunk(self, start, chunk_df):
        texts      = chunk_df[self.text_column].fillna('').tolist()
        all_scores = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            with torch.no_grad():
                batch_outputs = self.hf_pipeline(
                    batch, truncation=True, padding=True, max_length=512
                )
            batch_scores = [
                {item['label']: item['score'] for item in output}
                if isinstance(output, list)
                else {output['label']: output['score']}
                for output in batch_outputs
            ]
            all_scores.extend(batch_scores)

        chunk_df = chunk_df.copy()
        chunk_df["emotion_scores"] = all_scores
        end         = start + len(chunk_df)
        output_path = self.save_dir / f"{self.df_name}_{self.analysis_type}_chunk_{start}_{end}.parquet"
        chunk_df.to_parquet(output_path, index=False)
        print(f"[{datetime.now()}] ✅ Saved: {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline — Works
# ═══════════════════════════════════════════════════════════════════════════════

engine = create_engine('mysql+pymysql://andrew:$H0nggh0rzaq!@localhost:3306/mavenbookshelf')

# Persist the original start time across restarts so total runtime is accurate
_start_file = Path("pipeline_start.txt")
if _start_file.exists():
    pipeline_start = datetime.fromisoformat(_start_file.read_text().strip())
    print(f'Pipeline resumed — original start time: {pipeline_start}')
else:
    pipeline_start = datetime.now()
    _start_file.write_text(pipeline_start.isoformat())
    print('Pipeline started - Time:', pipeline_start)

print('Loading works - Time:', datetime.now())
works = pd.read_csv("./Data/goodreads_works.csv")

works['original_publication_year'] = (
    pd.to_numeric(works['original_publication_year'], errors='coerce').fillna(0).astype(int)
)
works['num_pages'] = (
    pd.to_numeric(works['num_pages'], errors='coerce').fillna(0).astype(int)
)
works['isbn13']      = works['isbn13'].astype('string').str.replace(".0", "", regex=False)
works['description'] = works['description'].fillna('')

logging.set_verbosity_error()

# Step 1: HF sentiment → chunk files
analyzer = ReviewAnalyzer(
    analysis_type="sentiment",
    model_name=MODEL_MAP["sentiment"],
    data_frame=works,
    save_dir="./sentiment_chunks",
    chunk_size=1000,
    batch_size=12,
    text_column="description",
    df_name="works",
)
analyzer.run()
analyzer.cleanup()
del analyzer
gc.collect()

# Step 2: VADER sentiment added to works df in-place.
# Must run BEFORE emotion analysis so the sentiment column is saved inside
# each emotion chunk file (making those files self-contained for the DB write).
works['sentiment'] = works['description'].apply(get_sentiment)
print('Works: VADER sentiment applied - Time:', datetime.now())

# Step 3: Emotion analysis → chunk files (each chunk contains all works columns
# + VADER sentiment + emotion_scores, so no need to re-join with works later)
analyzer = ReviewAnalyzer(
    analysis_type="emotion",
    model_name=MODEL_MAP["emotion"],
    data_frame=works,
    save_dir="./emotion_chunks",
    chunk_size=1000,
    batch_size=12,
    text_column="description",
    df_name="works",
)
analyzer.run()
analyzer.cleanup()
del analyzer, works
gc.collect()
print('Works: analysis complete, works df and models freed - Time:', datetime.now())

# Step 4: Stream chunk files → DB (peak memory: one chunk pair at a time)
write_to_db_streaming(
    df_name="works",
    engine=engine,
    table_name="works",
    chunk_size=1000,
    sentiment_dir="./sentiment_chunks",
    emotion_dir="./emotion_chunks",
)
print('Works: Wrote to MySQL Server - Time:', datetime.now())


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline — Reviews
# ═══════════════════════════════════════════════════════════════════════════════

print('Loading reviews - Time:', datetime.now())
reviews = pd.read_csv("./Data/goodreads_reviews.csv", low_memory=False)

# Step 1: HF sentiment → chunk files
analyzer = ReviewAnalyzer(
    analysis_type="sentiment",
    model_name=MODEL_MAP["sentiment"],
    data_frame=reviews,
    save_dir="./sentiment_chunks",
    chunk_size=10000,
    batch_size=16,
    text_column="review_text",
    df_name="reviews",
)
analyzer.run()
analyzer.cleanup()
del analyzer
gc.collect()

# Step 2: VADER sentiment added in-place (before emotion so it ends up in chunks)
reviews['sentiment'] = reviews['review_text'].fillna('').apply(get_sentiment)
print('Reviews: VADER sentiment applied - Time:', datetime.now())

# Step 3: Emotion analysis → chunk files
analyzer = ReviewAnalyzer(
    analysis_type="emotion",
    model_name=MODEL_MAP["emotion"],
    data_frame=reviews,
    save_dir="./emotion_chunks",
    chunk_size=10000,
    batch_size=16,
    text_column="review_text",
    df_name="reviews",
)
print('Reviews: Starting emotion analysis - Time:', datetime.now())
analyzer.run()
analyzer.cleanup()
del analyzer, reviews   # free ~1 GB before the DB write phase
gc.collect()
print('Reviews: analysis complete, reviews df and models freed - Time:', datetime.now())

# Step 4: Stream chunk files → DB (peak memory: one chunk pair at a time)
write_to_db_streaming(
    df_name="reviews",
    engine=engine,
    table_name="reviews",
    chunk_size=10000,
    sentiment_dir="./sentiment_chunks",
    emotion_dir="./emotion_chunks",
)

pipeline_end = datetime.now()
print('Reviews: Wrote to MySQL Database - Time:', pipeline_end)

elapsed = pipeline_end - pipeline_start
hours, remainder = divmod(int(elapsed.total_seconds()), 3600)
minutes, seconds = divmod(remainder, 60)
print(f'Total pipeline duration: {hours}h {minutes}m {seconds}s')
