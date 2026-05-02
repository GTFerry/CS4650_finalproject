import shutil
import subprocess
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

import pandas as pd
from huggingface_hub import snapshot_download
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data" / "raw"
HF = DATA / "earnings_call_hf"
MAEC = DATA / "MAEC"
PRICES = DATA / "stock_prices"

MAEC_REPO = "https://github.com/Earnings-Call-Dataset/MAEC-A-Multimodal-Aligned-Earnings-Conference-Call-Dataset-for-Financial-Risk-Prediction"


def fetch_hf():
    if (HF / "data" / "transcripts" / "train.txt").exists():
        print("[1/4] HF dataset already present.")
        return
    print("[1/4] Downloading HF earnings_call dataset...")
    snapshot_download(repo_id="jlh-ibm/earnings_call",
                      repo_type="dataset",
                      local_dir=str(HF),
                      ignore_patterns=["*.py", "*.zip"])


def fetch_maec():
    if (MAEC / "MAEC_Dataset").exists():
        print("[2/4] MAEC already present.")
        return
    print("[2/4] Cloning MAEC...")
    DATA.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth=1", MAEC_REPO, str(MAEC)], check=True)


def seed_prices():
    PRICES.mkdir(parents=True, exist_ok=True)
    source_price_directory = HF / "data" / "stock_prices"
    if not source_price_directory.exists():
        return
    copied_count = 0
    for source_csv in source_price_directory.glob("*.csv"):
        destination_csv = PRICES / source_csv.name
        if not destination_csv.exists():
            shutil.copy(source_csv, destination_csv)
            copied_count += 1
    print(f"[3/4] Seeded {copied_count} price CSVs from HF.")


def maec_ticker_dates():
    ticker_dates = defaultdict(list)
    for folder in (MAEC / "MAEC_Dataset").iterdir():
        if not folder.is_dir() or "_" not in folder.name:
            continue
        date_str, ticker = folder.name.split("_", 1)
        try:
            ticker_dates[ticker].append(pd.to_datetime(date_str, format="%Y%m%d"))
        except ValueError:
            pass
    return dict(ticker_dates)


def has_prices(ticker, dates):
    csv_path = PRICES / f"{ticker}.csv"
    if not csv_path.exists():
        return False
    try:
        price_frame = pd.read_csv(csv_path, parse_dates=["Date"])
        price_dates = pd.to_datetime(price_frame["Date"]).sort_values()
        return all(len(price_dates[price_dates >= call_date]) >= 2 for call_date in dates)
    except Exception:
        return False


def fetch_prices():
    import yfinance as yf
    ticker_dates = maec_ticker_dates()
    missing_tickers = [ticker for ticker, dates in ticker_dates.items() if not has_prices(ticker, dates)]
    if not missing_tickers:
        print("[4/4] All MAEC prices present.")
        return
    print(f"[4/4] yfinance: downloading {len(missing_tickers)} tickers...")
    failure_count = 0
    for ticker in tqdm(missing_tickers):
        dates = ticker_dates[ticker]
        start_date = (min(dates) - timedelta(days=10)).strftime("%Y-%m-%d")
        end_date = (max(dates) + timedelta(days=10)).strftime("%Y-%m-%d")
        try:
            price_frame = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)
            if price_frame.empty:
                failure_count += 1
                continue
            if isinstance(price_frame.columns, pd.MultiIndex):
                price_frame.columns = price_frame.columns.get_level_values(0)
            price_frame.index.name = "Date"
            price_frame.reset_index().to_csv(PRICES / f"{ticker}.csv", index=False)
        except Exception:
            failure_count += 1
    print(f"      Done. {len(missing_tickers) - failure_count} OK, {failure_count} failed (delisted).")


if __name__ == "__main__":
    DATA.mkdir(parents=True, exist_ok=True)
    fetch_hf()
    fetch_maec()
    seed_prices()
    fetch_prices()
    print("\nReady.")
