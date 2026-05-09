import sys

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

pd.set_option('display.max_columns', None)

# Definitions
MODEL_MAP = {
    "sentiment": "distilbert-base-uncased-finetuned-sst-2-english",
    "emotion": "j-hartmann/emotion-english-distilroberta-base"
}

OS="Linux" #Change to Windows for Windows Systems
LINUXPATH = "\home\andrew"
WINDOWSPATH = "D:\OneDrive - Insightful Tech Consulting"
SQLSERVER = "192.168.1.150"

#Functions
def process_saved_chunks(analysis_type, df_name):
    files = list(Path(f"{analysis_type}_chunks").glob(f"{df_name}_{analysis_type}_chunk_*.parquet"))
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
        self._analysis_type = analysis_type  # "sentiment" or "emotion"
        self._model_name = model_name
        self._data_frame = data_frame
        self._chunk_size = chunk_size
        self._batch_size = batch_size
        self._text_column = text_column
        self._df_name = df_name
        self._save_dir = Path(save_dir or f"{analysis_type}_chunks")
        self._save_dir.mkdir(exist_ok=True)
        self._use_quantized = use_quantized
        base_model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        if self._use_quantized:
            self.model = quantize_dynamic(base_model, {torch.nn.Linear}, dtype=torch.qint8)
            self.device = "cpu"
        else:
            self._model = base_model.to("cuda")
            self._device = "cuda"
        # Load model or pipeline
        if analysis_type == "sentiment":
            self.tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            if use_quantized:
                self._model = quantize_dynamic(base_model, {torch.nn.Linear}, dtype=torch.qint8)
                self._device = "cpu"
            else:
                self._model = base_model.to("cuda")
                self._device = "cuda"

            self._pipeline = None
        else:
            self._pipeline = pipeline("text-classification", model=self.model_name, device=0)
            self._tokenizer = None
            self._model = None

    def get_completed_chunks(self):
        return {
            int(f.stem.split("_")[-1])
            for f in self._save_dir.glob(f"{self._df_name}_{self._analysis_type}_chunk_*.parquet")
        }

    def chunk_dataframe(self):
        for start in range(0, len(self._data_frame), self._chunk_size):
            len(self._data_frame)
            yield start, self._data_frame.iloc[start:start + self._chunk_size]

    def run(self):
        completed = self.get_completed_chunks()
        for start, chunk_df in tqdm(self.chunk_dataframe(), desc=f"Running {self._analysis_type} analysis"):
            chunk_id = start // self._chunk_size
            if chunk_id in completed:
                continue
            print(f"[{datetime.now()}] Processing chunk {chunk_id}...")

            if self._analysis_type == "sentiment":
                self._process_sentiment_chunk(chunk_id, chunk_df,"work_id",self._device)
            else:
                chunk_df = chunk_df.copy()
                self._process_emotion_chunk(chunk_id, chunk_df)


    def _process_sentiment_chunk(self, chunk_id, chunk_df,key_column,device):
        texts = chunk_df[self._text_column].tolist()
        all_preds = []

        for i in range(0, len(texts), self._batch_size):
            batch = texts[i:i + self._batch_size]
            inputs = self._tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")

            if device == "cuda":
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            else:
                inputs = {
                    k: v.to(dtype=torch.float32, device="cpu") if v.dtype.is_floating_point else v.to("cpu")
                    for k, v in inputs.items()
                }

            if device == "cuda":
                with torch.autocast("cuda"), torch.no_grad():
                    outputs = self._model(**inputs)
                    preds = torch.argmax(outputs.logits, dim=1)
                    scores = torch.softmax(outputs.logits, dim=1)
            else:
                with torch.no_grad():
                    outputs = self._model(**inputs)
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
        result_df.to_parquet(self._save_dir / f"{self._df_name}_{self._analysis_type}_chunk_{chunk_id}.parquet")

    def _process_emotion_chunk(self, chunk_id, chunk_df):
        texts = chunk_df[self._text_column].tolist()
        _all_scores = []

        for i in range(0, len(texts), self._batch_size):
            batch = texts[i:i + self._batch_size]
            with torch.no_grad():
                batch_outputs = self._pipeline(
                    batch,
                    truncation=True,
                    padding=True,
                    max_length=512
                )

            for output in batch_outputs:
                if isinstance(output, dict):  # Single-label output
                    _all_scores.append({output['label']: output['score']})
                elif isinstance(output, list):  # Multi-label output
                    _all_scores.append({item['label']: item['score'] for item in output})
                else:
                    _all_scores.append({})

        assert len(_all_scores) == len(chunk_df), "Mismatch between scores and chunk size"
        chunk_df["emotion_scores"] = _all_scores
        output_path = self.save_dir / f"{self.df_name}_{self.analysis_type}_chunk_{chunk_id}.parquet"
        #output_path = self._save_dir / f"{self._df_name}_{self._analysis_type}_chunk_{start}_{end}.parquet"
        chunk_df.to_parquet(output_path, index=False)
        print(f"[{datetime.now()}] ✅ Saved: {output_path}")

    @property
    def analysis_type(self):
        return self._analysis_type

    @property
    def data_frame(self):
        return self._data_frame

    @property
    def df_name(self):
        return self._df_name

    @property
    def model_name(self):
        return self._model_name

    @property
    def chunk_size(self):
        return self._chunk_size

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def text_column(self):
        return self._text_column

    @property
    def save_dir(self):
        return self._save_dir

    @property
    def use_quantized(self):
        return self._use_quantized

    @analysis_type.setter
    def analysis_type(self, value):
        self._analysis_type = value

    @model_name.setter
    def model_name(self, value):
        self._model_name = value or "distilbert-base-uncased-finetuned-sst-2-english"

    @data_frame.setter
    def data_frame(self, value):
        self._data_frame = value

    @df_name.setter
    def df_name(self, value):
        self._df_name = value

    @chunk_size.setter
    def chunk_size(self, value):
        self._chunk_size = value

    @batch_size.setter
    def batch_size(self, value):
        self._batch_size = value

    @text_column.setter
    def text_column(self, value):
        self._text_column = value

    @save_dir.setter
    def save_dir(self, value):
        self._save_dir = value

    @use_quantized.setter
    def use_quantized(self,value):
        self._use_quantized = value


if OS == "Windows":
    works = pd.read_csv(fr"{WINDOWSPATH}\Data\Maven Bookshelf\goodreads_works.csv")
    reviews = pd.read_csv(fr"{WINDOWSPATH}\Data\Maven Bookshelf\goodreads_reviews.csv")
elif OS == "Linux":
    works = pd.read_csv(fr"{LINUXPATH}\Data\Maven Bookshelf\goodreads_works.csv")
    reviews = pd.read_csv(fr"{LINUXPATH}\Data\Maven Bookshelf\goodreads_reviews.csv")
else:
    print("Invalid OS")
    sys.exit(1)
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
chunks=process_saved_chunks(analyzer.analysis_type,
                            analyzer.df_name)


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
    batch_size=24,
    text_column="description",
    df_name = "works",
    use_quantized=False
)

analyzer.run()

chunks=process_saved_chunks(analyzer.analysis_type,
                            analyzer.df_name)


works_emotions = pd.concat(chunks, ignore_index=True)
# Assuming 'works' has a unique identifier like 'work_id'

works_combined = works.merge(
    works_sentiments[['work_id', 'label_hf', 'score_hf', 'sentiment_hf']],
    on='work_id',
    how='left'
).merge(
    works_emotions[['work_id', 'emotion_scores']],
    on='work_id',
    how='left'
)

works_combined['emotion_scores'] = works_combined['emotion_scores'].apply(json.dumps)

engine = create_engine(F'mysql+pymysql://root:$H0nggh0ri*@{SQLSERVER}:3306/mavenbookshelf')
works_combined.to_sql(
    'works',
    con=engine,
    if_exists='append',
    index=False,
        dtype={'emotion_scores': JSON}
)
print('Works: Wrote to mySQL Server - Time: ',datetime.now())
sys.exit(0)
print('Starting Reviews - Time: ',datetime.now())
analyzer = ReviewAnalyzer(
    analysis_type="sentiment",  # or "emotion"
    model_name=MODEL_MAP["sentiment"],
    data_frame=works,
    save_dir = r".\sentiment_chunks",
    chunk_size=1000,
    batch_size=12,
    text_column="description",
    df_name = "reviews",
    use_quantized=False
)

logging.set_verbosity_error() # Turn off logging except for major errors
analyzer.run()

reviews['sentiment'] = works['review_text'].apply(get_sentiment)
print('Reviews: Applied VADER Sentiment Analysis - Time: ',datetime.now())

chunks=process_saved_chunks(analyzer.analysis_type,
                            analyzer.df_name)

# Combine them
reviews_sentiments = pd.concat(chunks, ignore_index=True)

# Assuming 'works' has a unique identifier like 'work_id'
reviews_final = reviews.merge(
    reviews_sentiments[['work_id', 'label_hf', 'score_hf', 'sentiment_hf']],
    on='work_id',
    how='left'
)

analyzer = ReviewAnalyzer(
    analysis_type="emotion",
    model_name=MODEL_MAP["emotion"],
    data_frame=works,
    save_dir = r".\emotion_chunks",
    chunk_size=1000,
    batch_size=24,
    text_column="description",
    df_name = "reviews",
    use_quantized=False
)

analyzer.run()

chunks=process_saved_chunks(analyzer.analysis_type,
                            analyzer.df_name)


reviews_emotions = pd.concat(chunks, ignore_index=True)
# Assuming 'reviews' has a unique identifier like 'work_id'

reviews_combined = works.merge(
    reviews_sentiments[['work_id', 'label_hf', 'score_hf', 'sentiment_hf']],
    on='work_id',
    how='left'
).merge(
    reviews_emotions[['work_id', 'emotion_scores']],
    on='work_id',
    how='left'
)

reviews_combined['emotion_scores'] = reviews_combined['emotion_scores'].apply(json.dumps)

engine = create_engine('mysql+pymysql://root:$H0nggh0ri*@localhost:3306/mavenbookshelf')
reviews_combined.to_sql(
    'reviews',
    con=engine,
    if_exists='append',
    index=False,
        dtype={'emotion_scores': JSON}
)