from transformers import pipeline, logging
import pandas as pd
import torch, os, glob, json
from datetime import datetime
from sqlalchemy import create_engine
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tqdm import tqdm
from sqlalchemy.types import JSON
from pathlib import Path

# Definitions
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
SAVE_DIR = Path("sentiment_chunks")
SAVE_DIR.mkdir(exist_ok=True)


# Load sentiment classification pipeline
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    device=0,  # GPU
    truncation=True
)

# Load emotion classification pipeline
emotion_pipeline = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    tokenizer="j-hartmann/emotion-english-distilroberta-base",
    device=0,
    top_k=None,
    truncation=True,
    max_length=512
)

#Functions
def load_sentiment_model():
    return pipeline("sentiment-analysis", model=MODEL_NAME, device=0 if torch.cuda.is_available() else -1)

# 📦 Chunk Generator
def chunk_dataframe(df, chunk_size=1000):
    for c in range(0, len(df), chunk_size):
        yield c // chunk_size, df.iloc[c:c + chunk_size]

# 🧾 Check Completed Chunks
def get_completed_chunks():
    return {int(f.stem.split("_")[-1]) for f in SAVE_DIR.glob("sentiment_chunk_*.parquet")}

# 🚀 Run Sentiment Analysis on Chunk
def process_chunk(chunk_id, chunk_df, model):
    texts = chunk_df["review_text"].tolist()
    preds = model(texts)
    result_df = pd.DataFrame(preds)
    result_df["review_id"] = chunk_df["review_id"].values  # or use index if no ID
    result_df.to_parquet(SAVE_DIR / f"sentiment_chunk_{chunk_id}.parquet")

# 🧩 Main Runner
def run_sentiment_pipeline(df):
    model = load_sentiment_model()
    completed = get_completed_chunks()

    for chunk_id, chunk_df in chunk_dataframe(df, CHUNK_SIZE):
        if chunk_id in completed:
            continue
        print(f"Processing chunk {chunk_id}...")
        process_chunk(chunk_id, chunk_df, model)

# 🧬 Merge All Chunks
def merge_sentiment_chunks():
    files = sorted(SAVE_DIR.glob("sentiment_chunk_*.parquet"))
    dfs = [pd.read_parquet(f) for f in files]
    return pd.concat(dfs, ignore_index=True)


# ✅ Batched emotion score function
def batched_emotion_scores(texts, batch_size=16):
    all_scores = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        with torch.no_grad():
            batch_outputs = emotion_pipeline(batch)
        batch_scores = [
            {item['label']: item['score'] for item in output}
            for output in batch_outputs
        ]
        all_scores.extend(batch_scores)
    return all_scores

pd.set_option('display.max_colwidth', None)
print('Started at - Time: ',datetime.now())
# Read in the files
works = pd.read_csv("./Data/goodreads_works.csv")
reviews = pd.read_csv("./Data/goodreads_reviews.csv", low_memory=False)

# Here we convert some columns to integer and string values and fill missing values with 0
# This solves an error when casting a type with missing values

works['original_publication_year'] = pd.to_numeric(works['original_publication_year'], errors='coerce').fillna(0).astype(int)
works['num_pages'] = pd.to_numeric(works['num_pages'], errors='coerce').fillna(0).astype(int)
works['isbn13'] = works['isbn13'].astype('string')

# Remove the decimal point from the converted isbn13 column
works['isbn13'] =works['isbn13'].str.replace(".0","", regex=False)

# fill in missing values with blanks
works['description'] = works['description'].fillna('')

print('Works: Cleaned the data - Time: ',datetime.now())
# Perform Sentiment Analysis

# Create Vader Sentiment Score
analyzer=SentimentIntensityAnalyzer()

# define a function to get the score
def get_sentiment(text):
    return analyzer.polarity_scores(text)['compound']

# apply the function
works['sentiment'] = works['description'].apply(get_sentiment)
print('Works: Applied VADER Sentiment Analysis - Time: ',datetime.now())
# Now do a sentiment analysis using NLP

logging.set_verbosity_error() # Turn off logging except for major errors

# Define pipeline once


# Define batch size
batch_size = 24  # You can test 24 or 32 if memory allows

# Prepare texts
texts = works['description'].tolist()

# Run batched inference
results = []
for i in tqdm(range(0, len(texts), batch_size), desc="Sentiment Analysis"):
    batch = texts[i:i + batch_size]
    with torch.no_grad():
        batch_results = sentiment_analyzer(batch)
    results.extend(batch_results)

# Convert results to DataFrame
works['label_hf'] = [r['label'] for r in results]
works['score_hf'] = [r['score'] for r in results]
works['sentiment_hf'] = [
    r['score'] if r['label'] == 'POSITIVE' else -r['score']
    for r in results
]




print('Works: Applied Hugging Face Sentiment Analysis - Time: ',datetime.now())

process_and_save_chunks(
    df=works,
    text_column="description",
    chunk_size=1000,  # Small chunk for testing
    batch_size=12,
    output_dir="emotion_chunks",
    df_name="works"
)

# Load all processed chunks
files = glob.glob("emotion_chunks/works_emotions_*.parquet")
chunks = [pd.read_parquet(f) for f in sorted(files)]

# Combine them
works_emotions = pd.concat(chunks, ignore_index=True)

# Assuming 'works' has a unique identifier like 'work_id'
works_final = works.merge(
    works_emotions[['work_id', 'emotion_scores']],
    on='work_id',
    how='left'
)
works_final['emotion_scores'] = works_final['emotion_scores'].apply(json.dumps)

print('Works: Created Emotional Analysis - Time: ',datetime.now())
#print(works_final.head())
## Now write the data frame to the SQL Server
engine = create_engine('mysql+pymysql://root:$H0nggh0ri*@localhost:3306/mavenbookshelf')
works_final.to_sql(
    'works',
    con=engine,
    if_exists='replace',
    index=False,
        dtype={'emotion_scores': JSON}
)



print('Works: Wrote dataframe to MySQL - Time: ',datetime.now())


# Load emotion classification pipeline
#emotion_pipeline = pipeline(
#    "text-classification",
#    model="j-hartmann/emotion-english-distilroberta-base",
#    tokenizer="j-hartmann/emotion-english-distilroberta-base",
#    device=0,
#    top_k=None,
#    truncation=True,
#    max_length=512
#)

CHUNK_SIZE = 10000
SAVE_PATH = "sentiment_chunks/sentiment_results_chunk_{chunk_id}.parquet"




print('Reviews Sentiment Analysis Started At: ',datetime.now() )
run_sentiment_pipeline(df_reviews)
df_sentiment = merge_sentiment_chunks()


# Define batch size
batch_size = 24  # You can test 24 or 32 if memory allows

# Prepare texts
texts = reviews['review_text'].tolist()





for chunk_id in range(0, len(df), CHUNK_SIZE):
    if chunk_id // CHUNK_SIZE in completed_chunks:
        continue  # Skip already processed

    chunk = df.iloc[chunk_id : chunk_id + CHUNK_SIZE]
    results = run_sentiment_model(chunk["review_text"])  # Your inference logic
    results.to_parquet(SAVE_PATH.format(chunk_id=chunk_id // CHUNK_SIZE))



# Run batched inference
results = []
for i in tqdm(range(0, len(texts), batch_size), desc="Sentiment Analysis"):
    batch = texts[i:i + batch_size]
    with torch.no_grad():
        batch_results = sentiment_analyzer(batch)
    results.extend(batch_results)

# Convert results to DataFrame
reviews['label_hf'] = [r['label'] for r in results]
reviews['score_hf'] = [r['score'] for r in results]
reviews['sentiment_hf'] = [
    r['score'] if r['label'] == 'POSITIVE' else -r['score']
    for r in results
]

print('Reviews: Applied Hugging Face Sentiment Analysis - Time: ',datetime.now())

#reviews['emotion_scores'] = reviews['review_text'].apply(batched_emotion_scores)
#print('Reviews: Applied Emotion Score Analysis - Time: ',datetime.now())

#reviews.to_pickle("D:/OneDrive/Jupyter/My Projects/Maven Bookshelf Challenge/Models/reviews.pkl")

#print('Reviews: Pickled reviews model - Time: ',datetime.now())


# Apply to your dataset
#def get_emotion_scores(text):
#    scores = emotion_pipeline(text)[0]
#    return {item['label']: item['score'] for item in scores}

#works['emotion_scores'] = works['description'].apply(process_and_save_chunks(df, text_column, chunk_size=10000, batch_size=16, output_dir="emotion_chunks",df_name="reviews"):get_emotion_scores)
#print('Works: Applied Emotion Score Analysis - Time: ',datetime.now())

# Convert emotion scores to separate columns for emotion and score.

# Convert values into lists. This will be expanded in Power BI

# Pickle the works file
#works.to_pickle('D:/OneDrive/Jupyter/My Projects/Maven Bookshelf Challenge/Models/works.pkl')
#print('Works: Pickled Works Model - Time: ',datetime.now())
# Process the Reviews file

#print('Reviews: Started Processing - Time:', datetime.now())
# We will drop the user_id column, as it is not any use we just have a number and no names.
#reviews.drop([
#    "user_id","review_id"],
#    axis=1,
#    inplace=True
#)
#reviews["started_at"] = pd.to_datetime(reviews["started_at"], errors="coerce").dt.date
#reviews["read_at"] = pd.to_datetime(reviews["read_at"], errors="coerce").dt.date
#reviews["date_added"] = pd.to_datetime(reviews["read_at"], errors="coerce").dt.date
#print('Reviews: Cleaned the Data - Time: ',datetime.now())

# Create Vader Sentiment Score
#analyzer2=SentimentIntensityAnalyzer()

# define a function to get the score
#def get_sentiment(text):
#    return analyzer2.polarity_scores(text)['compound']

# apply the function
#reviews['sentiment'] = reviews['review_text'].apply(get_sentiment)
#print('Reviews: Applied VADER Sentiment Score - Time:',datetime.now())
# Now do a sentiment analysis using NLP
#logging.set_verbosity_error() # Turn off logging except for major errors

#sentiment_analyzer = pipeline("sentiment-analysis",
#                              model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
#                              device='cuda', # Use GPU
#                              truncation=True) # adding truncation here to truncate text before analyzing sentiment

#sentiment_scores = works['description'].apply(sentiment_analyzer) # Apply the Analyzer

# extract the label and score and create a sentiment score for all books
#reviews['label_hf'] = sentiment_scores.apply(lambda x: x[0]['label'])
#reviews['score_hf'] = sentiment_scores.apply(lambda x: x[0]['score'])

# This applies a negative value for a negative sentiment
#reviews['sentiment_hf'] = reviews.apply(lambda row: row['score_hf'] if row['label_hf'] == 'POSITIVE' else -row['score_hf'], axis=1)
#print('Reviews: Applied Hugging Face Sentiment Score - Time: ', datetime.now())

# Write the reviews data frame to a MySql table
#reviews.to_sql('reviews', con=engine, if_exists='replace', index=False)


# ✅ Load the emotion classification pipeline with GPU (CUDA)
#emotion_pipeline2 = pipeline(
#    "text-classification",
#    model="j-hartmann/emotion-english-distilroberta-base",
#    tokenizer="j-hartmann/emotion-english-distilroberta-base",
#    device=0,  # 0 = CUDA GPU
#    truncation=True,
#    max_length=512,
#    top_k=None
#)

# 🛠️ Example usage
# Load your reviews DataFrame (assumed already available)
# reviews = pd.read_csv("your_reviews_data.csv")

#reviews['emotion_scores'] = reviews['review_text'].apply(batched_emotion_scores)
#print('Reviews: Applied Emotion Score Analysis - Time: ',datetime.now())

#reviews.to_pickle("D:/OneDrive/Jupyter/My Projects/Maven Bookshelf Challenge/Models/reviews.pkl")

#print('Reviews: Pickled reviews model - Time: ',datetime.now())