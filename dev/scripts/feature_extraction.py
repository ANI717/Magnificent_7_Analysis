import argparse
import pandas as pd
import os


def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def compute_macd(close):
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist


def compute_bollinger_bands(close, window=20):
    sma = close.rolling(window=window).mean()
    std = close.rolling(window=window).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    return upper, sma, lower


def add_optional_features(df):
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)

    # Date/time features
    df['day_of_week'] = df.index.dayofweek

    # Price-based volatility
    df['High_Low'] = df['High'] - df['Low']
    df['Close_Open'] = df['Close'] - df['Open']

    # Volume trend
    df['Volume_rolling_mean_5'] = df['Volume'].rolling(5).mean()

    # Bollinger band position
    df['BB_pct'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])

    # Candle position
    df['candle_pos'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])

    # Z-score of close
    mean = df['Close'].rolling(window=20).mean()
    std = df['Close'].rolling(window=20).std()
    df['Close_zscore'] = (df['Close'] - mean) / std

    # Percentile rank
    df['Close_pct_rank_20'] = df['Close'].rolling(window=20).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)

    return df


def extract_features(df):
    # Target
    df["return"] = df["Close"].pct_change().shift(-1)

    # Lag features
    for lag in range(1, 6):
        df[f"Close_lag_{lag}"] = df["Close"].shift(lag)

    # Rolling features
    df["Close_rolling_mean_5"] = df["Close"].rolling(window=5).mean()

    # RSI
    df["RSI"] = compute_rsi(df["Close"])

    # MACD
    df["MACD"], df["MACD_signal"], df["MACD_hist"] = compute_macd(df["Close"])

    # Bollinger Bands
    df["BB_upper"], df["BB_middle"], df["BB_lower"] = compute_bollinger_bands(df["Close"])

    # Add optional features
    df = add_optional_features(df)

    # Clean NaNs
    df.dropna(inplace=True)

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", required=True, help="Stock ticker symbol (e.g., NVDA)")
    parser.add_argument("--location", help="Input CSV path (default: ../data/{symbol}.csv)")
    args = parser.parse_args()

    symbol = args.symbol.upper()
    location = args.location or f"../data/{symbol}.csv"

    if not os.path.exists(location):
        raise FileNotFoundError(f"File not found: {location}")

    df = pd.read_csv(location, parse_dates=True, index_col=0)

    df = extract_features(df)

    df.to_csv(location)
    print(f"✅ Features extracted and saved to {location} with {df.shape[0]} rows")


if __name__ == "__main__":
    main()
