import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib


# -----------------------------
# Feature helpers
# -----------------------------
def compute_rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period).mean()
    ma_down = down.ewm(alpha=1/period).mean()
    rs = ma_up / (ma_down + 1e-9)
    return 100 - (100 / (1 + rs))


def compute_macd(series):
    ema12 = series.ewm(span=12).mean()
    ema26 = series.ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    hist = macd - signal
    return macd, signal, hist


def build_features_and_targets(df: pd.DataFrame):
    """
    Build X (features) and y (% returns) from OHLCV data for ONE ticker.
    """
    closes = df["Close"].astype(float).values
    volumes = df["Volume"].astype(float).values

    s = pd.Series(closes)
    X = []
    y = []

    # use last 60 candles to predict next-day % return
    for i in range(60, len(closes)):
        window = s[i-60:i]
        last_close = closes[i-1]
        today_close = closes[i]

        # target: percentage return from yesterday to today
        ret = (today_close - last_close) / last_close

        lag1 = float(window.iloc[-1])
        ma7 = float(window.tail(7).mean())
        ma21 = float(window.tail(21).mean())
        ema12 = float(window.ewm(span=12).mean().iloc[-1])
        ema26 = float(window.ewm(span=26).mean().iloc[-1])
        ema50 = float(window.ewm(span=50).mean().iloc[-1])
        std21 = float(window.tail(21).std())
        upper_bb = ma21 + 2 * std21
        lower_bb = ma21 - 2 * std21
        rsi14 = float(compute_rsi(window).iloc[-1])
        macd_v, macd_s, macd_h = compute_macd(window)

        macd_val = float(macd_v.iloc[-1])
        macd_sig_val = float(macd_s.iloc[-1])
        macd_hist_val = float(macd_h.iloc[-1])

        last_volume = float(volumes[i-1])

        X.append([
            lag1, ma7, ma21,
            ema12, ema26, ema50,
            std21, upper_bb, lower_bb,
            rsi14, macd_val, macd_sig_val, macd_hist_val,
            last_volume
        ])
        y.append(ret)

    return np.array(X), np.array(y)


# -----------------------------
# MAIN TRAINING SCRIPT
# -----------------------------
if __name__ == "__main__":
    # Option B: Indian + US stocks
    TICKERS = [
        # Indian (NSE)
        "RELIANCE.NS",
        "TCS.NS",
        "INFY.NS",
        "HDFCBANK.NS",
        "ICICIBANK.NS",
        "SBIN.NS",

        # US
        "AAPL",
        "MSFT",
        "GOOGL",
        "TSLA",
        "NVDA",
    ]

    X_all = []
    y_all = []

    print("📥 Downloading & building dataset...")
    for ticker in TICKERS:
        print(f"   -> {ticker}")
        df = yf.download(ticker, period="5y", interval="1d", progress=False)

        if df.empty or len(df) < 100:
            print("      (no / too little data, skipping)")
            continue

        X_t, y_t = build_features_and_targets(df)
        if len(X_t) == 0:
            print("      (no samples after feature building, skipping)")
            continue

        X_all.append(X_t)
        y_all.append(y_t)

    if not X_all:
        raise RuntimeError("No data collected for any ticker!")

    X = np.vstack(X_all)
    y = np.concatenate(y_all)

    print("✅ Final dataset shape:", X.shape, "targets:", y.shape)

    # scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # train RandomForest on % returns
    print("🌲 Training RandomForestRegressor...")
    rf = RandomForestRegressor(
        n_estimators=400,
        max_depth=20,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_scaled, y)

    # save model + scaler in ROOT (no folders)
    joblib.dump(rf, "rf_model.joblib")
    joblib.dump(scaler, "scaler_feat.joblib")

    print("🎉 Done! Saved rf_model.joblib and scaler_feat.joblib in project root.")
