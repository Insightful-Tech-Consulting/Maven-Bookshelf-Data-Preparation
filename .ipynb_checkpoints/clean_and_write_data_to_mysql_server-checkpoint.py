from transformers import pipeline
import pandas as pd
import torch
from datetime import datetime
from sqlalchemy import create_engine
from transformers import pipeline, logging
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os

# ✅ Load the emotion classification pipeline with GPU (CUDA)
emotion_pipeline2 = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    tokenizer="j-hartmann/emotion-english-distilroberta-base",
    device=0,  # 0 = CUDA GPU
    truncation=True,
    max_length=512,
    top_k=None
)


# ✅ Batched emotion score function
def batched_emotion_scores(texts, batch_size=16):
    all_scores = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        with torch.no_grad():
            batch_outputs = emotion_pipeline2(batch)
        batch_scores = [
            {item['label']: item['score'] for item in output}
            for output in batch_outputs
        ]
        all_scores.extend(batch_scores)
    return all_scores


# ✅ Process & save chunk-wise
def process_and_save_chunks(df, text_column, chunk_size=10000, batch_size=16, output_dir="emotion_chunks"):
    os.makedirs(output_dir, exist_ok=True)
    total_rows = len(df)

    for start in range(0, total_rows, chunk_size):
        end = min(start + chunk_size, total_rows)
        chunk = df.iloc[start:end].copy()

        print(f"[{datetime.now()}] Processing reviews {start} to {end}...")

        chunk_scores = batched_emotion_scores(chunk[text_column].tolist(), batch_size=batch_size)
        chunk['emotion_scores'] = chunk_scores

        output_path = os.path.join(output_dir, f"review_emotions_{start}_{end}.parquet")
        chunk.to_parquet(output_path, index=False)

        print(f"[{datetime.now()}] ✅ Saved: {output_path}")


# 🛠️ Example usage
# Load your reviews DataFrame (assumed already available)
# reviews = pd.read_csv("your_reviews_data.csv")

process_and_save_chunks(
    df=reviews,
    text_column="review_text",
    chunk_size=10000,
    batch_size=16,
    output_dir="emotion_chunks"
)


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

sentiment_analyzer = pipeline("sentiment-analysis",
                              model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
                              device='cuda', # Use GPU
                              truncation=True) # adding truncation here to truncate text before analyzing sentiment

sentiment_scores = works['description'].apply(sentiment_analyzer) # Apply the Analyser

# extract the label and score and create a sentiment score for all books
works['label_hf'] = sentiment_scores.apply(lambda x: x[0]['label'])
works['score_hf'] = sentiment_scores.apply(lambda x: x[0]['score'])

# This applies a negative value for a negative sentiment
works['sentiment_hf'] = works.apply(lambda row: row['score_hf'] if row['label_hf'] == 'POSITIVE' else -row['score_hf'], axis=1)

print('Works: Applied Hugging Face Sentiment Analysis - Time: ',datetime.now())
# ## Now write the data frame to the SQL Server
engine = create_engine('mysql+pymysql://root:$H0nggh0ri*@localhost:3306/mavenbookshelf')
works.to_sql('works', con=engine, if_exists='replace', index=False)
print('Works: Wrote dataframe to MySQL - Time: ',datetime.now())

# Load emotion classification pipeline
emotion_pipeline = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    tokenizer="j-hartmann/emotion-english-distilroberta-base",
    top_k=None,
    truncation=True,
    max_length=512
)


# Apply to your dataset
def get_emotion_scores(text):
    scores = emotion_pipeline(text)[0]
    return {item['label']: item['score'] for item in scores}

works['emotion_scores'] = works['description'].apply(get_emotion_scores)
print('Works: Applied Emotion Score Analysis - Time: ',datetime.now())
# Convert emotion scores to separate columns for emotion and score.

# Convert values into lists. This will be expanded in Power BI

# Pickle the works file
works.to_pickle('D:/OneDrive/Jupyter/My Projects/Maven Bookshelf Challenge/Models/works.pkl')
print('Works: Pickled Works Model - Time: ',datetime.now())
# Process the Reviews file

print('Reviews: Started Processing - Time:', datetime.now())
# We will drop the user_id column, as it is not any use we just have a number and no names.
reviews.drop([
    "user_id","review_id"], 
    axis=1, 
    inplace=True
)
reviews["started_at"] = pd.to_datetime(reviews["started_at"], errors="coerce").dt.date
reviews["read_at"] = pd.to_datetime(reviews["read_at"], errors="coerce").dt.date
reviews["date_added"] = pd.to_datetime(reviews["read_at"], errors="coerce").dt.date
print('Reviews: Cleaned the Data - Time: ',datetime.now())

# Create Vader Sentiment Score
analyzer2=SentimentIntensityAnalyzer()

# define a function to get the score
def get_sentiment(text):
    return analyzer2.polarity_scores(text)['compound']

# apply the function
reviews['sentiment'] = reviews['review_text'].apply(get_sentiment)
print('Reviews: Applied VADER Sentiment Score - Time:',datetime.now())
# Now do a sentiment analysis using NLP

from transformers import pipeline, logging

logging.set_verbosity_error() # Turn off logging except for major errors

sentiment_analyzer = pipeline("sentiment-analysis",
                              model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
                              device='cuda', # Use GPU
                              truncation=True) # adding truncation here to truncate text before analyzing sentiment

sentiment_scores = works['description'].apply(sentiment_analyzer) # Apply the Analyser

# extract the label and score and create a sentiment score for all books
reviews['label_hf'] = sentiment_scores.apply(lambda x: x[0]['label'])
reviews['score_hf'] = sentiment_scores.apply(lambda x: x[0]['score'])

# This applies a negative value for a negative sentiment
reviews['sentiment_hf'] = reviews.apply(lambda row: row['score_hf'] if row['label_hf'] == 'POSITIVE' else -row['score_hf'], axis=1)
print('Reviews: Applied Hugging Face Sentiment Score - Time: ', datetime.now())

# Write the reviews data frame to a MySql table
reviews.to_sql('reviews', con=engine, if_exists='replace', index=False)

from transformers import pipeline

# Load emotion classification pipeline
emotion_pipeline2 = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    tokenizer="j-hartmann/emotion-english-distilroberta-base",
    top_k=None,
    truncation=True,
    max_length=512
)

# Apply to your dataset
def get_emotion_scores(text):
    scores = emotion_pipeline2(text)[0]
    return {item['label']: item['score'] for item in scores}

reviews['emotion_scores'] = reviews['review_text'].apply(get_emotion_scores)
print('Reviews: Applied Emotion Score Analysis - Time: ',datetime.now())

reviews.to_pickle('D:/OneDrive/Jupyter/My Projects/Maven Bookshelf Challenge/Models/reviews.pkl')

print('Reviews: Pickled reviews model - Time: ',datetime.now())