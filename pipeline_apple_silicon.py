from transformers import pipeline, logging, AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import torch, json, gc, contextlib, csv
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


# ── Pipeline Logger ───────────────────────────────────────────────────────────
class PipelineLogger:
    """
    Writes a structured CSV log of every pipeline event for post-run analysis
    and visualisation (e.g. throughput time-series, thermal throttle detection).

    CSV columns
    -----------
    timestamp       ISO-8601 wall-clock time of the event
    event           PIPELINE_START | PIPELINE_END
                    STAGE_START    | STAGE_END
                    CHUNK_START    | CHUNK_COMPLETE | CHUNK_SKIPPED
    dataset         works | reviews
    stage           sentiment | vader | emotion | db_write
    chunk_id        zero-based chunk index within the stage
    records_start   index of the first record in the chunk (inclusive)
    records_end     index of the last record in the chunk (exclusive)
    duration_s      wall-clock seconds elapsed for this event (where applicable)
    notes           free-text detail (e.g. human-readable total duration)

    Usage
    -----
    Pair CHUNK_COMPLETE rows with their CHUNK_START rows on
    (dataset, stage, chunk_id) to derive per-chunk throughput:
        records_per_second = (records_end - records_start) / duration_s
    """

    HEADER = [
        'timestamp', 'event', 'dataset', 'stage',
        'chunk_id', 'records_start', 'records_end', 'duration_s', 'notes'
    ]

    def __init__(self, log_dir='.'):
        run_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_path = Path(log_dir) / f'pipeline_log_{run_ts}.csv'
        self._stage_starts = {}
        self._chunk_starts = {}
        with open(self.log_path, 'w', newline='') as f:
            csv.writer(f).writerow(self.HEADER)
        print(f'📋 Log: {self.log_path}')

    # ── internal helpers ──────────────────────────────────────────────────────
    def _write(self, event, dataset='', stage='', chunk_id='',
               records_start='', records_end='', duration_s='', notes=''):
        with open(self.log_path, 'a', newline='') as f:
            csv.writer(f).writerow([
                datetime.now().isoformat(),
                event, dataset, stage,
                chunk_id, records_start, records_end, duration_s, notes
            ])

    def _elapsed(self, key, store):
        t0 = store.get(key)
        return round((datetime.now() - t0).total_seconds(), 2) if t0 else ''

    # ── public API ────────────────────────────────────────────────────────────
    def pipeline_start(self, t0: datetime):
        self._write('PIPELINE_START', notes=t0.isoformat())

    def pipeline_end(self, t0: datetime):
        secs = (datetime.now() - t0).total_seconds()
        h, r = divmod(int(secs), 3600)
        m, s = divmod(r, 60)
        self._write('PIPELINE_END', duration_s=round(secs, 2),
                    notes=f'{h}h {m}m {s}s')

    def stage_start(self, dataset, stage):
        self._stage_starts[(dataset, stage)] = datetime.now()
        self._write('STAGE_START', dataset=dataset, stage=stage)

    def stage_end(self, dataset, stage):
        self._write('STAGE_END', dataset=dataset, stage=stage,
                    duration_s=self._elapsed((dataset, stage), self._stage_starts))

    def chunk_start(self, dataset, stage, chunk_id, records_start):
        self._chunk_starts[(dataset, stage, chunk_id)] = datetime.now()
        self._write('CHUNK_START', dataset=dataset, stage=stage,
                    chunk_id=chunk_id, records_start=records_start)

    def chunk_complete(self, dataset, stage, chunk_id, records_start, records_end):
        key = (dataset, stage, chunk_id)
        self._write('CHUNK_COMPLETE', dataset=dataset, stage=stage,
                    chunk_id=chunk_id, records_start=records_start,
                    records_end=records_end,
                    duration_s=self._elapsed(key, self._chunk_starts))

    def chunk_skipped(self, dataset, stage, chunk_id):
        self._write('CHUNK_SKIPPED', dataset=dataset, stage=stage,
                    chunk_id=chunk_id, notes='checkpoint — already complete')


# ── Streaming DB writer ───────────────────────────────────────────────────────
def write_to_db_streaming(df_name, engine, table_name, chunk_size,
                          sentiment_dir='sentiment_chunks',
                          emotion_dir='emotion_chunks',
                          logger=None):
    """
    Write a dataset to the DB by streaming through saved emotion chunk files
    one at a time. Each emotion chunk already contains the full row data plus
    VADER sentiment. We merge HF sentiment from the matching sentiment chunk
    then write to the database.

    Peak memory per iteration: ~one emotion chunk + one sentiment chunk.

    Join key: reviews → review_id (unique per row, avoids many-to-many blowup)
              works   → work_id
    """
    sentiment_path = Path(sentiment_dir)
    emotion_path   = Path(emotion_dir)
    join_key = 'review_id' if df_name == 'reviews' else 'work_id'

    emotion_files = sorted(emotion_path.glob(f'{df_name}_emotion_chunk_*.parquet'))
    if not emotion_files:
        print(f'⚠️  No emotion chunks found for {df_name} in {emotion_dir}')
        return

    if logger:
        logger.stage_start(df_name, 'db_write')

    for i, ef in enumerate(tqdm(emotion_files, desc=f'Writing {df_name} → {table_name}')):
        parts    = ef.stem.split('_')
        start    = int(parts[-2])
        end      = int(parts[-1])
        chunk_id = start // chunk_size

        if logger:
            logger.chunk_start(df_name, 'db_write', chunk_id, start)

        emot_df = pd.read_parquet(ef)

        sf = sentiment_path / f'{df_name}_sentiment_chunk_{chunk_id}.parquet'
        if sf.exists():
            sent_df = pd.read_parquet(sf)
            sent_df = sent_df.drop_duplicates(subset=join_key)
            emot_df = emot_df.merge(
                sent_df[[join_key, 'label_hf', 'score_hf', 'sentiment_hf']],
                on=join_key, how='left'
            )
            del sent_df
        else:
            print(f'⚠️  Sentiment chunk missing for chunk_id={chunk_id}, skipping HF merge')

        emot_df['emotion_scores'] = emot_df['emotion_scores'].apply(json.dumps)

        # First chunk recreates the table with the correct schema;
        # subsequent chunks append to it.
        if_exists = 'replace' if i == 0 else 'append'
        emot_df.to_sql(
            table_name, con=engine, if_exists=if_exists,
            index=False, dtype={'emotion_scores': JSON}, chunksize=500,
        )

        if logger:
            logger.chunk_complete(df_name, 'db_write', chunk_id, start, end)

        del emot_df
        gc.collect()

    if logger:
        logger.stage_end(df_name, 'db_write')

    print(f'✅ {df_name} → {table_name} complete')


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
        text_column='review_text',
        df_name='reviews',
        use_quantized=False,
        device=None,
        logger=None,
    ):
        self.analysis_type = analysis_type
        self.model_name    = model_name or 'distilbert-base-uncased-finetuned-sst-2-english'
        self.data_frame    = data_frame
        self.chunk_size    = chunk_size
        self.batch_size    = batch_size
        self.text_column   = text_column
        self.df_name       = df_name
        self.save_dir      = Path(save_dir or f'{analysis_type}_chunks')
        self.save_dir.mkdir(exist_ok=True)
        self.logger        = logger

        if use_quantized:
            self.device = 'cpu'
        else:
            self.device = device or DEVICE

        print(f'ReviewAnalyzer [{analysis_type}] using device: {self.device}')

        base_model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

        if analysis_type == 'sentiment':
            self.tokenizer   = AutoTokenizer.from_pretrained(self.model_name)
            self.model       = (
                quantize_dynamic(base_model, {torch.nn.Linear}, dtype=torch.qint8)
                if use_quantized
                else base_model.to(self.device)
            )
            self.hf_pipeline = None
        else:
            pipeline_device  = self.device if self.device != 'cuda' else 0
            self.hf_pipeline = pipeline(
                'text-classification',
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
        print(f'✅ [{self.analysis_type}] model released from memory')

    # ── Checkpoint helpers ────────────────────────────────────────────────────
    def get_completed_chunks(self):
        completed = set()
        for f in self.save_dir.glob(f'{self.df_name}_{self.analysis_type}_chunk_*.parquet'):
            parts = f.stem.split('_')
            try:
                if self.analysis_type == 'emotion':
                    completed.add(int(parts[-2]) // self.chunk_size)
                else:
                    completed.add(int(parts[-1]))
            except (ValueError, IndexError):
                pass
        return completed

    def chunk_dataframe(self):
        for start in range(0, len(self.data_frame), self.chunk_size):
            yield start, self.data_frame.iloc[start:start + self.chunk_size]

    # ── Main entry point ──────────────────────────────────────────────────────
    def run(self):
        if self.logger:
            self.logger.stage_start(self.df_name, self.analysis_type)

        completed = self.get_completed_chunks()
        for start, chunk_df in tqdm(
            self.chunk_dataframe(), desc=f'Running {self.analysis_type} analysis'
        ):
            chunk_id = start // self.chunk_size
            if chunk_id in completed:
                if self.logger:
                    self.logger.chunk_skipped(self.df_name, self.analysis_type, chunk_id)
                continue

            if self.logger:
                self.logger.chunk_start(self.df_name, self.analysis_type, chunk_id, start)

            print(f'[{datetime.now()}] Processing chunk {chunk_id}...')

            if self.analysis_type == 'sentiment':
                self._process_sentiment_chunk(chunk_id, chunk_df, self.device)
            else:
                self._process_emotion_chunk(start, chunk_df)

            if self.logger:
                self.logger.chunk_complete(
                    self.df_name, self.analysis_type, chunk_id,
                    start, start + len(chunk_df)
                )

        if self.logger:
            self.logger.stage_end(self.df_name, self.analysis_type)

    # ── Sentiment chunk ───────────────────────────────────────────────────────
    def _process_sentiment_chunk(self, chunk_id, chunk_df, device):
        texts     = chunk_df[self.text_column].fillna('').tolist()
        all_preds = []

        for i in range(0, len(texts), self.batch_size):
            batch  = texts[i:i + self.batch_size]
            inputs = self.tokenizer(
                batch, padding=True, truncation=True, max_length=512, return_tensors='pt'
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            if device == 'cuda':
                ctx = torch.autocast('cuda')
            elif device == 'cpu':
                ctx = torch.autocast('cpu')
            else:
                ctx = contextlib.nullcontext()   # MPS: autocast not supported

            with ctx, torch.no_grad():
                outputs = self.model(**inputs)
                preds   = torch.argmax(outputs.logits, dim=1)
                scores  = torch.softmax(outputs.logits, dim=1)

            for pred, score in zip(preds, scores):
                label      = 'POSITIVE' if pred.item() == 1 else 'NEGATIVE'
                confidence = score[pred].item()
                all_preds.append({
                    'label_hf':     label,
                    'score_hf':     confidence,
                    'sentiment_hf': confidence if label == 'POSITIVE' else -confidence,
                })

        result_df            = pd.DataFrame(all_preds)
        result_df['work_id'] = chunk_df['work_id'].values
        if 'review_id' in chunk_df.columns:
            result_df['review_id'] = chunk_df['review_id'].values

        result_df.to_parquet(
            self.save_dir / f'{self.df_name}_{self.analysis_type}_chunk_{chunk_id}.parquet'
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
        chunk_df['emotion_scores'] = all_scores
        end         = start + len(chunk_df)
        output_path = self.save_dir / f'{self.df_name}_emotion_chunk_{start}_{end}.parquet'
        chunk_df.to_parquet(output_path, index=False)
        print(f'[{datetime.now()}] ✅ Saved: {output_path}')


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline — setup
# ═══════════════════════════════════════════════════════════════════════════════

# ── Database name ─────────────────────────────────────────────────────────────
# Change this to match the target database for each machine before running.
# Create the empty database first if it doesn't already exist:
#   CREATE DATABASE mavenbookshelf_macbook;
DB_NAME = 'mavenbookshelf_macbook'

engine = create_engine(f'mysql+pymysql://andrew:$H0nggh0rzaq!@localhost:3306/{DB_NAME}')

# Persist start time across restarts so total runtime is always accurate.
# Delete pipeline_start.txt manually if you want a completely fresh clock.
_start_file = Path('pipeline_start.txt')
if _start_file.exists():
    pipeline_start = datetime.fromisoformat(_start_file.read_text().strip())
    print(f'Pipeline resumed — original start time: {pipeline_start}')
else:
    pipeline_start = datetime.now()
    _start_file.write_text(pipeline_start.isoformat())
    print('Pipeline started - Time:', pipeline_start)

# Log file is named with the actual wall-clock start of this process invocation
# so each restart creates its own log; the PIPELINE_START row records the true
# original start time for accurate total-duration calculation.
logger = PipelineLogger(log_dir=str(Path(__file__).parent))
logger.pipeline_start(pipeline_start)

logging.set_verbosity_error()


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline — Works
# ═══════════════════════════════════════════════════════════════════════════════

print('Loading works - Time:', datetime.now())
works = pd.read_csv('./Data/goodreads_works.csv')

works['original_publication_year'] = (
    pd.to_numeric(works['original_publication_year'], errors='coerce').fillna(0).astype(int)
)
works['num_pages'] = (
    pd.to_numeric(works['num_pages'], errors='coerce').fillna(0).astype(int)
)
works['isbn13']      = works['isbn13'].astype('string').str.replace('.0', '', regex=False)
works['description'] = works['description'].fillna('')

# Step 1: HF sentiment → chunk files
analyzer = ReviewAnalyzer(
    analysis_type='sentiment',
    model_name=MODEL_MAP['sentiment'],
    data_frame=works,
    save_dir='./sentiment_chunks',
    chunk_size=1000,
    batch_size=12,
    text_column='description',
    df_name='works',
    logger=logger,
)
analyzer.run()
analyzer.cleanup()
del analyzer
gc.collect()

# Step 2: VADER sentiment
logger.stage_start('works', 'vader')
works['sentiment'] = works['description'].apply(get_sentiment)
logger.stage_end('works', 'vader')
print('Works: VADER sentiment applied - Time:', datetime.now())

# Step 3: Emotion analysis → chunk files
analyzer = ReviewAnalyzer(
    analysis_type='emotion',
    model_name=MODEL_MAP['emotion'],
    data_frame=works,
    save_dir='./emotion_chunks',
    chunk_size=1000,
    batch_size=12,
    text_column='description',
    df_name='works',
    logger=logger,
)
analyzer.run()
analyzer.cleanup()
del analyzer, works
gc.collect()
print('Works: analysis complete, works df and models freed - Time:', datetime.now())

# Step 4: Stream chunk files → DB
write_to_db_streaming(
    df_name='works',
    engine=engine,
    table_name='works',
    chunk_size=1000,
    sentiment_dir='./sentiment_chunks',
    emotion_dir='./emotion_chunks',
    logger=logger,
)
print('Works: Wrote to MySQL Server - Time:', datetime.now())


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline — Reviews
# ═══════════════════════════════════════════════════════════════════════════════

print('Loading reviews - Time:', datetime.now())
reviews = pd.read_csv('./Data/goodreads_reviews.csv', low_memory=False)

# Step 1: HF sentiment → chunk files
analyzer = ReviewAnalyzer(
    analysis_type='sentiment',
    model_name=MODEL_MAP['sentiment'],
    data_frame=reviews,
    save_dir='./sentiment_chunks',
    chunk_size=10000,
    batch_size=8,
    text_column='review_text',
    df_name='reviews',
    logger=logger,
)
analyzer.run()
analyzer.cleanup()
del analyzer
gc.collect()

# Step 2: VADER sentiment
logger.stage_start('reviews', 'vader')
reviews['sentiment'] = reviews['review_text'].fillna('').apply(get_sentiment)
logger.stage_end('reviews', 'vader')
print('Reviews: VADER sentiment applied - Time:', datetime.now())

# Step 3: Emotion analysis → chunk files
analyzer = ReviewAnalyzer(
    analysis_type='emotion',
    model_name=MODEL_MAP['emotion'],
    data_frame=reviews,
    save_dir='./emotion_chunks',
    chunk_size=10000,
    batch_size=8,
    text_column='review_text',
    df_name='reviews',
    logger=logger,
)
print('Reviews: Starting emotion analysis - Time:', datetime.now())
analyzer.run()
analyzer.cleanup()
del analyzer, reviews
gc.collect()
print('Reviews: analysis complete, reviews df and models freed - Time:', datetime.now())

# Step 4: Stream chunk files → DB
write_to_db_streaming(
    df_name='reviews',
    engine=engine,
    table_name='reviews',
    chunk_size=10000,
    sentiment_dir='./sentiment_chunks',
    emotion_dir='./emotion_chunks',
    logger=logger,
)

pipeline_end = datetime.now()
logger.pipeline_end(pipeline_start)

# Clean up start-time file on successful completion so the next run starts fresh
_start_file.unlink(missing_ok=True)

print('Reviews: Wrote to MySQL Database - Time:', pipeline_end)
elapsed = pipeline_end - pipeline_start
hours, remainder = divmod(int(elapsed.total_seconds()), 3600)
minutes, seconds = divmod(remainder, 60)
print(f'Total pipeline duration: {hours}h {minutes}m {seconds}s')
