from transformers import pipeline, logging, AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import torch, glob, json
from torch.quantization import quantize_dynamic
from datetime import datetime
from sqlalchemy import create_engine
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tqdm import tqdm
from sqlalchemy.types import JSON
from pathlib import Path

#pd.set_option('display.max_columns', None)

# Definitions
MODEL_MAP = {
    "sentiment": "distilbert-base-uncased-finetuned-sst-2-english",
    "emotion": "j-hartmann/emotion-english-distilroberta-base"
}
def chunk_dataframe(df, chunk_size: None):
    """
    Splits a DataFrame into sequential chunks of a specified size, yielding each with its index.

    This generator function divides the input DataFrame into non-overlapping chunks, yielding
    a tuple of (chunk_index, chunk_df) for each segment. Useful for batch processing, progress
    tracking, and restartable workflows in large-scale data pipelines.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame to be chunked.
    chunk_size : int or None
        Number of rows per chunk. If None, defaults to 10,000.

    Yields
    ------
    Tuple[int, pandas.DataFrame]
        A tuple containing the chunk index (starting from 0) and the corresponding DataFrame slice.

    Notes
    -----
    - Chunk index is computed as `c // chunk_size` for compatibility with resume logic.
    - Final chunk may contain fewer rows if `len(df)` is not divisible by `chunk_size`.
    """
    if chunk_size is None:
        chunk_size = 10000

    for c in range(0, len(df), chunk_size):
        yield c // chunk_size, df.iloc[c:c + chunk_size]

# 🧬 Merge All Chunks
def merge_sentiment_chunks(save_dir,analysis_type):
    files = sorted(save_dir.glob(f"{analysis_type}_chunk_*.parquet"))
    dfs = [pd.read_parquet(f) for f in files]
    return pd.concat(dfs, ignore_index=True)

# ✅ Batched emotion score function
def batched_scores(texts, analysis_pipeline, batch_size=16):
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
    files = list(Path(f"{analysis_type}_chunks").glob(f"{df_name}_{analysis_type}_chunk_*.parquet"))
    print(f"Looking for files in: {analysis_type}_chunks")
    print(f"Pattern: {df_name}_{analysis_type}_chunk_*.parquet")
    print(f"Matched files: {[f.name for f in files]}")
    file_chunks = []
    for f in sorted(files):
        try:
            file_chunks.append(pd.read_parquet(f))
        except Exception as e:
            print(f"⚠️ Skipped {f.name}: {e}")
    return file_chunks

class ReviewAnalyzer:
    def __init__(self, analysis_type, model_name=None, data_frame=None, save_dir=None,
                 chunk_size=1000, batch_size=16, text_column="review_text", df_name="reviews", use_quantized=False):
        self.analysis_type = analysis_type  # "sentiment" or "emotion"
        self.model_name = model_name or "distilbert-base-uncased-finetuned-sst-2-english"
        self.data_frame = data_frame
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.text_column = text_column
        self.df_name = df_name
        self.save_dir = Path(save_dir or f"{analysis_type}_chunks")
        self.save_dir.mkdir(exist_ok=True)
        base_model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        if use_quantized:
            self.model = quantize_dynamic(base_model, {torch.nn.Linear}, dtype=torch.qint8)
            self.device = "cpu"
        else:
            self.model = base_model.to("cuda")
            self.device = "cuda"
        # Load model or pipeline
        if analysis_type == "sentiment":
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if use_quantized:
                self.model = quantize_dynamic(base_model, {torch.nn.Linear}, dtype=torch.qint8)
                self.device = "cpu"
            else:
                self.model = base_model.to("cuda")
                self.device = "cuda"

            self.pipeline = None
        else:
            self.pipeline = pipeline("text-classification", model=self.model_name, device=0)
            self.tokenizer = None
            self.model = None

    def get_completed_chunks(self):
        return {
            int(f.stem.split("_")[-1])
            for f in self.save_dir.glob(f"{self.df_name}_{self.analysis_type}_chunk_*.parquet")
        }

    def chunk_dataframe(self):
        for start in range(0, len(self.data_frame), self.chunk_size):
            yield start, self.data_frame.iloc[start:start + self.chunk_size]

    def run(self):
        completed = self.get_completed_chunks()
        for start, chunk_df in tqdm(self.chunk_dataframe(), desc=f"Running {self.analysis_type} analysis"):
            chunk_id = start // self.chunk_size
            if chunk_id in completed:
                continue
            print(f"[{datetime.now()}] Processing chunk {chunk_id}...")

            if self.analysis_type == "sentiment":
                self._process_sentiment_chunk(chunk_id, chunk_df,"work_id",self.device)
            else:
                self._process_emotion_chunk(start, chunk_df)


    def _process_sentiment_chunk(self, chunk_id, chunk_df,key_column,device):
        texts = chunk_df[self.text_column].tolist()
        all_preds = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")

            if device == "cuda":
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            else:
                inputs = {
                    k: v.to(dtype=torch.float32, device="cpu") if v.dtype.is_floating_point else v.to("cpu")
                    for k, v in inputs.items()
                }

            if device == "cuda":
                with torch.autocast("cuda"), torch.no_grad():
                    outputs = self.model(**inputs)
                    preds = torch.argmax(outputs.logits, dim=1)
                    scores = torch.softmax(outputs.logits, dim=1)
            else:
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    preds = torch.argmax(outputs.logits, dim=1)
                    scores = torch.softmax(outputs.logits, dim=1)

            for pred, score in zip(preds, scores):
                label = "POSITIVE" if pred.item() == 1 else "NEGATIVE"
                confidence = score[pred].item()
                sentiment = confidence if label == "POSITIVE" else -confidence
                all_preds.append({
                    "label_hf": label,
                    "score_hf": confidence,
                    "sentiment_hf": sentiment
                })

        result_df = pd.DataFrame(all_preds)
        result_df[key_column] = chunk_df[key_column].values
        result_df.to_parquet(self.save_dir / f"{self.df_name}_{self.analysis_type}_chunk_{chunk_id}.parquet")

    def _process_emotion_chunk(self, start, chunk_df):
        texts = chunk_df[self.text_column].tolist()
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

        chunk_df["emotion_scores"] = all_scores
        end = start + len(chunk_df)
        output_path = self.save_dir / f"{self.df_name}_{self.analysis_type}_chunk_{start}_{end}.parquet"
        chunk_df.to_parquet(output_path, index=False)
        print(f"[{datetime.now()}] ✅ Saved: {output_path}")

    def get_analysis_type(self):
        return self.analysis_type

    def get_data_frame(self):
        return self.data_frame

    def get_df_name(self):
        return self.df_name

works = pd.read_csv(r"D:\OneDrive - Insightful Tech Consulting\Data\Maven Bookshelf\goodreads_works.csv")

# Here we convert some columns to integer and string values and fill missing values with 0
# This solves an error when casting a type with missing values
works['original_publication_year'] = pd.to_numeric(works['original_publication_year'], errors='coerce').fillna(0).astype(int)
works['num_pages'] = pd.to_numeric(works['num_pages'], errors='coerce').fillna(0).astype(int)
works['isbn13'] = works['isbn13'].astype('string').str.replace(".0","", regex=False)


# Before we start we will fill missing values with a blank string.
works['description'] = works['description'].fillna('')

analyzer = ReviewAnalyzer(
    analysis_type="sentiment",  # or "emotion"
    model_name=MODEL_MAP["sentiment"],
    data_frame=works,
    save_dir = r".\sentiment_chunks",
    chunk_size=1000,
    batch_size=12,
    text_column="description",
    df_name = "works",
    use_quantized=False
)

logging.set_verbosity_error() # Turn off logging except for major errors
analyzer.run()

# Create Vader Sentiment Score
vader_analyzer=SentimentIntensityAnalyzer()
def get_sentiment(text):
    return vader_analyzer.polarity_scores(text)['compound']

# apply the function
works['sentiment'] = works['description'].apply(get_sentiment)
print('Works: Applied VADER Sentiment Analysis - Time: ',datetime.now())
chunks=process_saved_chunks(analyzer.get_analysis_type(),
                            analyzer.get_df_name())


# Combine them
works_sentiments = pd.concat(chunks, ignore_index=True)

# Assuming 'works' has a unique identifier like 'work_id'
works_final = works.merge(
    works_sentiments[['work_id', 'label_hf', 'score_hf', 'sentiment_hf']],
    on='work_id',
    how='left'
)

analyzer = ReviewAnalyzer(
    analysis_type="emotion",
    model_name=MODEL_MAP["emotion"],
    data_frame=works,
    save_dir = r".\emotion_chunks",
    chunk_size=1000,
    batch_size=12,
    text_column="description",
    df_name = "works",
    use_quantized=False
)
chunks=process_saved_chunks(analyzer.get_analysis_type(),
                            analyzer.get_df_name())


works_emotions = pd.concat(chunks, ignore_index=True)
# Assuming 'works' has a unique identifier like 'work_id'
works_emotions_final = works.merge(
    works_emotions[['work_id', 'emotion_scores']],
    on='work_id',
    how='left'
)
works_emotions_final['emotion_scores'] = works_emotions['emotion_scores'].apply(json.dumps)

#works_final['emotion_scores'] = works_final['sentiment'].apply(json.dumps)

engine = create_engine('mysql+pymysql://root:<password>@localhost:3306/mavenbookshelf')
works_final.to_sql(
    'works',
    con=engine,
    if_exists='replace',
    index=False,
        dtype={'emotion_scores': JSON}
)
print('Works: Wrote to mySQL Server - Time: ',datetime.now())

