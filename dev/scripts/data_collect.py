import argparse
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os


def validate_interval(start: str, end: str, interval: str) -> str:
    start_date = datetime.strptime(start, "%Y-%m-%d")
    end_date = datetime.strptime(end, "%Y-%m-%d")
    today = datetime.today()

    if interval in ["1m", "2m", "5m", "15m", "30m", "60m"]:
        max_range = timedelta(days=59)
        if end_date > today or (today - start_date > max_range):
            print(f"⚠️ WARNING: {interval} data is only available for the last 60 days.")
            print("👉 Switching to '1d' interval instead.")
            return "1d"
    return interval


def fetch_data(symbol: str, start: str, end: str, interval: str) -> pd.DataFrame:
    ticker = yf.Ticker(symbol)
    all_data = []

    start_date = datetime.strptime(start, "%Y-%m-%d")
    end_date = datetime.strptime(end, "%Y-%m-%d")

    while start_date < end_date:
        chunk_end = min(start_date + timedelta(days=59), end_date)
        print(f"Fetching: {start_date.date()} to {chunk_end.date()}")

        data = ticker.history(start=start_date, end=chunk_end, interval=interval)
        if not data.empty:
            all_data.append(data)

        start_date = chunk_end

    if not all_data:
        raise ValueError("No data fetched. Check your symbol, date range, or interval.")

    df = pd.concat(all_data)
    df = df[~df.index.duplicated(keep='first')]
    df.sort_index(inplace=True)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", required=True, help="Stock ticker symbol (e.g., NVDA)")
    parser.add_argument("--start", default="2020-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=datetime.today().strftime("%Y-%m-%d"), help="End date (YYYY-MM-DD)")
    parser.add_argument("--interval", default="1d", help="Interval (e.g., 1d, 15m, 1h)")
    parser.add_argument("--location", help="Output file path (default: ../data/{symbol}.csv)")

    args = parser.parse_args()

    # Default save path
    if not args.location:
        args.location = f"../data/{args.symbol.upper()}.csv"

    # Validate interval and adjust if needed
    safe_interval = validate_interval(args.start, args.end, args.interval)

    print(f"\n🟢 Downloading {args.symbol} from {args.start} to {args.end} with interval: {safe_interval}")

    df = fetch_data(args.symbol, args.start, args.end, safe_interval)

    os.makedirs(os.path.dirname(args.location), exist_ok=True)
    df.to_csv(args.location)
    print(f"✅ Saved {len(df)} rows to {args.location}")


if __name__ == "__main__":
    main()
