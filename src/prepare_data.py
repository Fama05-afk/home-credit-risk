import duckdb
from pathlib import Path
from sklearn.model_selection import train_test_split

PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "target"
ID_COL = "loan_id"

con = duckdb.connect("data/dbt_output/dev.duckdb", read_only=True)
df  = con.execute("SELECT * FROM mart_credit_features").df()
con.close()

print(f"Shape: {df.shape}")

df = df.drop(columns=[ID_COL])

train_full, test = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df[TARGET],
)

train, val = train_test_split(
    train_full, test_size=0.125, random_state=42, stratify=train_full[TARGET],
)

print(f"Train: {len(train):,} · Val: {len(val):,} · Test: {len(test):,}")
print(f"Train target rate: {train[TARGET].mean():.4f}")
print(f"Val   target rate: {val[TARGET].mean():.4f}")
print(f"Test  target rate: {test[TARGET].mean():.4f}")

train.to_parquet(PROCESSED_DIR / "train.parquet", index=False)
val.to_parquet(PROCESSED_DIR   / "val.parquet",   index=False)
test.to_parquet(PROCESSED_DIR  / "test.parquet",  index=False)