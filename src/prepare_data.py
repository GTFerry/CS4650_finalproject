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
HF   = DATA / "earnings_call_hf"
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
    src = HF / "data" / "stock_prices"
    if not src.exists():
        return
    n = 0
    for csv in src.glob("*.csv"):
        dst = PRICES / csv.name
        if not dst.exists():
            shutil.copy(csv, dst)
            n += 1
    print(f"[3/4] Seeded {n} price CSVs from HF.")


def maec_ticker_dates():
    out = defaultdict(list)
    for f in (MAEC / "MAEC_Dataset").iterdir():
        if not f.is_dir() or "_" not in f.name:
            continue
        date_str, ticker = f.name.split("_", 1)
        try:
            out[ticker].append(pd.to_datetime(date_str, format="%Y%m%d"))
        except ValueError:
            pass
    return dict(out)


def has_prices(ticker, dates):
    csv = PRICES / f"{ticker}.csv"
    if not csv.exists():
        return False
    try:
        df = pd.read_csv(csv, parse_dates=["Date"])
        d = pd.to_datetime(df["Date"]).sort_values()
        return all(len(d[d >= x]) >= 2 for x in dates)
    except Exception:
        return False


def fetch_prices():
    import yfinance as yf
    td = maec_ticker_dates()
    todo = [t for t, d in td.items() if not has_prices(t, d)]
    if not todo:
        print("[4/4] All MAEC prices present.")
        return
    print(f"[4/4] yfinance: downloading {len(todo)} tickers...")
    fail = 0
    for t in tqdm(todo):
        dates = td[t]
        start = (min(dates) - timedelta(days=10)).strftime("%Y-%m-%d")
        end = (max(dates) + timedelta(days=10)).strftime("%Y-%m-%d")
        try:
            df = yf.download(t, start=start, end=end, auto_adjust=True, progress=False)
            if df.empty:
                fail += 1
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.index.name = "Date"
            df.reset_index().to_csv(PRICES / f"{t}.csv", index=False)
        except Exception:
            fail += 1
    print(f"      Done. {len(todo) - fail} OK, {fail} failed (delisted).")


if __name__ == "__main__":
    DATA.mkdir(parents=True, exist_ok=True)
    fetch_hf()
    fetch_maec()
    seed_prices()
    fetch_prices()
    print("\nReady.")
