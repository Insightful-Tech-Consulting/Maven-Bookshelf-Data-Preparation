import pandas as pd

# Load CSV with flexible error handling
df = pd.read_csv(
    "cleaned_goodreads.csv",
    quoting=1,
    on_bad_lines='skip',
    encoding='utf-8',
    engine='python'
)

# Drop rows that are fully empty (just to tidy things up)
df.dropna(how='all', inplace=True)

# Filter rows to those with exactly 20 columns
df_clean = df[df.apply(lambda row: len(row) == 20, axis=1)]

# Export cleaned file
df_clean.to_csv("final_goodreads.csv", index=False, quoting=1)
