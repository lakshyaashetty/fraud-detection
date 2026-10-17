import pandas as pd

RAW = "data/raw/PS_20174392719_1491204439457_log.csv"
OUT = "data/raw/transactions.parquet"

DTYPES = {
    "step": "int16", "type": "category", "amount": "float64",
    "nameOrig": "string", "nameDest": "string",
    "isFraud": "int8", "isFlaggedFraud": "int8",
}


def build():
    df = pd.read_csv(RAW, usecols=list(DTYPES), dtype=DTYPES)  # balance columns never loaded
    df = df[df["type"].isin(["TRANSFER", "CASH_OUT"])].reset_index(drop=True)
    df["type"] = df["type"].astype(str)
    df.to_parquet(OUT, index=False)
    print(df.shape, "fraud rate:", round(df["isFraud"].mean(), 5))


def load() -> pd.DataFrame:
    return pd.read_parquet(OUT)


if __name__ == "__main__":
    build()
