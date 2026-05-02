import os
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


def get_price_frame_for_ticker(ticker):
    if ticker in _price_cache:
        return _price_cache[ticker]

    price_csv_path = PRICE_DIR / f"{ticker}.csv"
    if not price_csv_path.exists():
        _price_cache[ticker] = None
        return None

    price_frame = pd.read_csv(price_csv_path, parse_dates=["Date"]).set_index("Date").sort_index()
    _price_cache[ticker] = price_frame
    return price_frame


def next_day_return(ticker, call_date):
    price_frame = get_price_frame_for_ticker(ticker)
    if price_frame is None:
        return None

    price_rows_after_call = price_frame[price_frame.index >= call_date]
    if len(price_rows_after_call) < 2:
        return None

    first_close = price_rows_after_call.iloc[0]["Close"]
    second_close = price_rows_after_call.iloc[1]["Close"]
    return float((second_close - first_close) / first_close)


def financial_features(ticker, call_date):
    nan = float("nan")
    price_frame = get_price_frame_for_ticker(ticker)
    if price_frame is None:
        return [nan, nan, nan]

    closing_prices_before_call = price_frame[price_frame.index < call_date]["Close"]
    if len(closing_prices_before_call) < 31:
        return [nan, nan, nan]

    return_5_day = float(
        (closing_prices_before_call.iloc[-1] - closing_prices_before_call.iloc[-6]) / closing_prices_before_call.iloc[-6]
    )
    return_30_day = float(
        (closing_prices_before_call.iloc[-1] - closing_prices_before_call.iloc[-31]) / closing_prices_before_call.iloc[-31]
    )
    daily_returns = closing_prices_before_call.pct_change().dropna()
    annualized_volatility = float(daily_returns.iloc[-30:].std() * np.sqrt(252))
    return [return_5_day, return_30_day, annualized_volatility]


def label_for(ret):
    for i in range(NUM_CLASSES):
        if RETURN_BINS[i] <= ret < RETURN_BINS[i + 1]:
            return i
    return NUM_CLASSES - 1


def _record(ticker, date, text, source):
    if len(text.strip()) < MIN_TEXT_CHARS:
        return None
    next_day_ret = next_day_return(ticker, date)
    if next_day_ret is None:
        return None

    return {
        "ticker": ticker,
        "date": date,
        "text": text,
        "return": next_day_ret,
        "label": label_for(next_day_ret),
        "source": source,
        "financial": financial_features(ticker, date),
    }


def load_hf():
    records = []
    transcript_dir = HF_DIR / "transcripts"
    if not transcript_dir.exists():
        return records

    for ticker_directory in sorted(transcript_dir.iterdir()):
        if not ticker_directory.is_dir():
            continue
        ticker = ticker_directory.name
        for transcript_file in sorted(ticker_directory.glob("*.txt")):
            filename_parts = transcript_file.stem.split("-")
            try:
                call_date = pd.to_datetime("-".join(filename_parts[:3]), format="%Y-%b-%d")
            except ValueError:
                continue
            record = _record(ticker, call_date, transcript_file.read_text(), "hf")
            if record:
                records.append(record)
    return records


def load_maec():
    records = []
    if not MAEC_DIR.exists():
        return records

    for folder in sorted(MAEC_DIR.iterdir()):
        if not folder.is_dir() or "_" not in folder.name:
            continue
        date_str, ticker = folder.name.split("_", 1)
        try:
            call_date = pd.to_datetime(date_str, format="%Y%m%d")
        except ValueError:
            continue
        text_file = folder / "text.txt"
        if not text_file.exists():
            continue
        record = _record(ticker, call_date, text_file.read_text(), "maec")
        if record:
            records.append(record)
    return records


def load_parquet(path, source):
    if not path.exists():
        return []
    records = []
    parquet_frame = pd.read_parquet(path)
    for _, row in parquet_frame.iterrows():
        ticker = str(row["ticker"]).upper().strip()
        call_date = pd.to_datetime(row["call_date"]).normalize()
        transcript_text = str(row["text"]) if not pd.isna(row["text"]) else ""
        record = _record(ticker, call_date, transcript_text, source)
        if record:
            records.append(record)
    return records


def load_all():
    hf = load_hf()
    kurry = load_parquet(KURRY_PATH, "kurry")
    glopardo = load_parquet(GLOPARDO_PATH, "glopardo")
    maec = load_maec()
    seen = set()
    records = []
    for source in [hf, kurry, glopardo, maec]:
        for record in source:
            key = (record["ticker"], record["date"].date())
            if key not in seen:
                seen.add(key)
                records.append(record)
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
