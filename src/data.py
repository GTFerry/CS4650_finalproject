import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
from torch.utils.data import Dataset

nltk.download("punkt_tab", quiet=True)

DATA_DIR = Path(os.environ.get("DATA_DIR", "data/raw"))
HF_DIR = DATA_DIR / "earnings_call_hf" / "data"
MAEC_DIR = DATA_DIR / "MAEC" / "MAEC_Dataset"
PRICE_DIR = DATA_DIR / "stock_prices"
KURRY_PATH = DATA_DIR / "kurry.parquet"
GLOPARDO_PATH = DATA_DIR / "glopardo.parquet"

NUM_CLASSES = 3
RETURN_BINS = [-float("inf"), -0.01, 0.01, float("inf")]

SPLIT_DATES = {
    "train": (pd.Timestamp("1900-01-01"), pd.Timestamp("2021-01-01")),
    "val": (pd.Timestamp("2021-01-01"), pd.Timestamp("2023-01-01")),
    "test": (pd.Timestamp("2023-01-01"), pd.Timestamp("2099-01-01")),
}

MIN_TEXT_CHARS = 500
MAX_SENTS = 600

_price_cache = {}

def split_sentences(text):
    return sent_tokenize(text)[:MAX_SENTS]


def load_prices(ticker):
    if ticker not in _price_cache:
        csv = PRICE_DIR / f"{ticker}.csv"
        if not csv.exists():
            _price_cache[ticker] = None
        else:
            df = pd.read_csv(csv, parse_dates=["Date"]).set_index("Date").sort_index()
            _price_cache[ticker] = df
    return _price_cache[ticker]


def next_day_return(ticker, call_date):
    df = load_prices(ticker)
    if df is None:
        return None
    after = df[df.index >= call_date]
    if len(after) < 2:
        return None
    a, b = after.iloc[0]["Close"], after.iloc[1]["Close"]
    return float((b - a) / a)


def financial_features(ticker, call_date):
    nan = float("nan")
    df = load_prices(ticker)
    if df is None:
        return [nan, nan, nan]
    before = df[df.index < call_date]["Close"]
    if len(before) < 31:
        return [nan, nan, nan]
    r5  = float((before.iloc[-1] - before.iloc[-6])  / before.iloc[-6])
    r30 = float((before.iloc[-1] - before.iloc[-31]) / before.iloc[-31])
    daily = before.pct_change().dropna()
    vol = float(daily.iloc[-30:].std() * np.sqrt(252))
    return [r5, r30, vol]


def label_for(ret):
    for i in range(NUM_CLASSES):
        if RETURN_BINS[i] <= ret < RETURN_BINS[i + 1]:
            return i
    return NUM_CLASSES - 1


def _record(ticker, date, text, source):
    if len(text.strip()) < MIN_TEXT_CHARS:
        return None
    ret = next_day_return(ticker, date)
    if ret is None:
        return None
    return {
        "ticker": ticker,
        "date": date,
        "text": text,
        "return": ret,
        "label": label_for(ret),
        "source": source,
        "financial": financial_features(ticker, date),
    }


def load_hf():
    out = []
    transcript_dir = HF_DIR / "transcripts"
    if not transcript_dir.exists():
        return out
    for ticker_dir in sorted(transcript_dir.iterdir()):
        if not ticker_dir.is_dir():
            continue
        ticker = ticker_dir.name
        for f in sorted(ticker_dir.glob("*.txt")):
            parts = f.stem.split("-")
            try:
                date = pd.to_datetime("-".join(parts[:3]), format="%Y-%b-%d")
            except ValueError:
                continue
            r = _record(ticker, date, f.read_text(), "hf")
            if r:
                out.append(r)
    return out


def load_maec():
    out = []
    if not MAEC_DIR.exists():
        return out
    for folder in sorted(MAEC_DIR.iterdir()):
        if not folder.is_dir() or "_" not in folder.name:
            continue
        date_str, ticker = folder.name.split("_", 1)
        try:
            date = pd.to_datetime(date_str, format="%Y%m%d")
        except ValueError:
            continue
        text_file = folder / "text.txt"
        if not text_file.exists():
            continue
        r = _record(ticker, date, text_file.read_text(), "maec")
        if r:
            out.append(r)
    return out


def load_parquet(path, source):
    if not path.exists():
        return []
    out = []
    df = pd.read_parquet(path)
    for _, row in df.iterrows():
        ticker = str(row["ticker"]).upper().strip()
        date = pd.to_datetime(row["call_date"]).normalize()
        text = str(row["text"]) if not pd.isna(row["text"]) else ""
        r = _record(ticker, date, text, source)
        if r:
            out.append(r)
    return out


def load_all():
    hf       = load_hf()
    kurry    = load_parquet(KURRY_PATH, "kurry")
    glopardo = load_parquet(GLOPARDO_PATH, "glopardo")
    maec     = load_maec()
    seen = set()
    records = []
    for source in [hf, kurry, glopardo, maec]:
        for r in source:
            key = (r["ticker"], r["date"].date())
            if key not in seen:
                seen.add(key)
                records.append(r)
    records.sort(key=lambda r: r["date"])
    return records


def split(records, name):
    start, end = SPLIT_DATES[name]
    return [r for r in records if start <= r["date"] < end]


class EarningsDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        return self.records[i]
