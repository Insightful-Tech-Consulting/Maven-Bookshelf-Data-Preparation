import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.types import JSON
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration — update connection string as needed
# ═══════════════════════════════════════════════════════════════════════════════
ENGINE_URL = 'mysql+pymysql://andrew:$H0nggh0rzaq!@localhost:3306/mavenbookshelf'

WORKS_CSV    = './Data/goodreads_works.csv'
REVIEWS_CSV  = './Data/goodreads_reviews.csv'


# ═══════════════════════════════════════════════════════════════════════════════
# Utility
# ═══════════════════════════════════════════════════════════════════════════════
def load_chunks(analysis_type, df_name):
    """Load and concatenate all saved chunk parquet files."""
    files = sorted(Path(f"{analysis_type}_chunks").glob(
        f"{df_name}_{analysis_type}_chunk_*.parquet"
    ))
    print(f"Loading {len(files)} {df_name} {analysis_type} chunks...")
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_parquet(f))
        except Exception as e:
            print(f"⚠️  Skipped {f.name}: {e}")
    return pd.concat(dfs, ignore_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Works pipeline
# ═══════════════════════════════════════════════════════════════════════════════
print('Loading works - Time:', datetime.now())
works = pd.read_csv(WORKS_CSV)
works['original_publication_year'] = (
    pd.to_numeric(works['original_publication_year'], errors='coerce').fillna(0).astype(int)
)
works['num_pages'] = (
    pd.to_numeric(works['num_pages'], errors='coerce').fillna(0).astype(int)
)
works['isbn13']      = works['isbn13'].astype('string').str.replace(".0", "", regex=False)
works['description'] = works['description'].fillna('')

# VADER sentiment on works
vader_analyzer = SentimentIntensityAnalyzer()
works['sentiment'] = works['description'].apply(
    lambda t: vader_analyzer.polarity_scores(str(t))['compound']
)
print('Works: Applied VADER Sentiment - Time:', datetime.now())

# Load HF sentiment chunks and merge on work_id
works_sentiments = load_chunks('sentiment', 'works')
works_final = works.merge(
    works_sentiments[['work_id', 'label_hf', 'score_hf', 'sentiment_hf']],
    on='work_id',
    how='left'
)

# Load emotion chunks — emotion chunks contain full row data so just concat
works_emotions = load_chunks('emotion', 'works')
works_final = works_final.merge(
    works_emotions[['work_id', 'emotion_scores']],
    on='work_id',
    how='left'
)
works_final['emotion_scores'] = works_final['emotion_scores'].apply(json.dumps)

print('Works shape:', works_final.shape)

# Write works to database
engine = create_engine(ENGINE_URL)
works_final.to_sql(
    'works',
    con=engine,
    if_exists='replace',
    index=False,
    schema='mavenbookshelf',
    dtype={'emotion_scores': JSON}
)
print('Works: Wrote to MySQL - Time:', datetime.now())


# ═══════════════════════════════════════════════════════════════════════════════
# Reviews pipeline
# ═══════════════════════════════════════════════════════════════════════════════
print('Loading reviews - Time:', datetime.now())
reviews = pd.read_csv(REVIEWS_CSV, low_memory=False)

# VADER sentiment on reviews
reviews['sentiment'] = reviews['review_text'].apply(
    lambda t: vader_analyzer.polarity_scores(str(t))['compound']
)
print('Reviews: Applied VADER Sentiment - Time:', datetime.now())

# Load HF sentiment chunks
# Sentiment chunks contain: review_id (or work_id), label_hf, score_hf, sentiment_hf
# Merge on review_id to preserve one score per review
reviews_sentiments = load_chunks('sentiment', 'reviews')
print('Reviews sentiments shape:', reviews_sentiments.shape)

# Check what key column is available in sentiment chunks
print('Reviews sentiments columns:', reviews_sentiments.columns.tolist())

# Merge sentiment — use review_id if present, fall back to index alignment
if 'review_id' in reviews_sentiments.columns:
    reviews_combined = reviews.merge(
        reviews_sentiments[['review_id', 'label_hf', 'score_hf', 'sentiment_hf']],
        on='review_id',
        how='left'
    )
else:
    # Sentiment chunks were saved with work_id only — align by position
    print('⚠️  review_id not found in sentiment chunks — aligning by position')
    reviews_combined = reviews.copy()
    reviews_combined['label_hf']     = reviews_sentiments['label_hf'].values
    reviews_combined['score_hf']     = reviews_sentiments['score_hf'].values
    reviews_combined['sentiment_hf'] = reviews_sentiments['sentiment_hf'].values

print('After sentiment merge:', reviews_combined.shape)

# Load emotion chunks — emotion chunks contain full review row data
reviews_emotions = load_chunks('emotion', 'reviews')
print('Reviews emotions shape:', reviews_emotions.shape)
print('Reviews emotions columns:', reviews_emotions.columns.tolist())

# Merge emotion — use review_id if present, fall back to position alignment
if 'review_id' in reviews_emotions.columns:
    reviews_combined = reviews_combined.merge(
        reviews_emotions[['review_id', 'emotion_scores']],
        on='review_id',
        how='left'
    )
else:
    print('⚠️  review_id not found in emotion chunks — aligning by position')
    reviews_combined['emotion_scores'] = reviews_emotions['emotion_scores'].values

print('After emotion merge:', reviews_combined.shape)

reviews_combined['emotion_scores'] = reviews_combined['emotion_scores'].apply(json.dumps)

# Write reviews to database
reviews_combined.to_sql(
    'reviews',
    con=engine,
    if_exists='replace',
    index=False,
    schema='mavenbookshelf',
    dtype={'emotion_scores': JSON}
)
print('Reviews: Wrote to MySQL - Time:', datetime.now())
print('All done!')
