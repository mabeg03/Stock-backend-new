import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib


# --------------------------------------
# Indicators
# --------------------------------------
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


# --------------------------------------
# Build features + target for one stock
# --------------------------------------
def build_features(df):
    closes = df["Close"].values
    volumes = df["Volume"].values
    s = pd.Series(closes)

    X, y = [], []

    for i in range(60, len(closes)):
        window = s[i-60:i]
        last_close = closes[i-1]
        today_close = closes[i]

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
        macd_sig = float(macd_s.iloc[-1])
        macd_hist = float(macd_h.iloc[-1])

        last_volume = float(volumes[i-1])

        X.append([
            lag1, ma7, ma21,
            ema12, ema26, ema50,
            std21, upper_bb, lower_bb,
            rsi14, macd_val, macd_sig, macd_hist,
            last_volume
        ])

        y.append(ret)

    return np.array(X), np.array(y)


# --------------------------------------
# MAIN TRAINER
# --------------------------------------
if __name__ == "__main__":

    TICKERS = [
        # Indian high liquidity
        "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
        "SBIN.NS", "KOTAKBANK.NS", "BHARTIARTL.NS", "ITC.NS", "ASIANPAINT.NS",

        # US tech & mega caps
        "AAPL", "MSFT", "GOOGL", "AMZN", "META",
        "TSLA", "NVDA", "NFLX", "JPM", "V"
    ]

    X_all, Y_all = [], []

    print("📥 Downloading data...")
    for t in TICKERS:
        print(" →", t)
        df = yf.download(t, period="5y", interval="1d", progress=False)

        if df.empty or len(df) < 100:
            print("   Skipped (no data)")
            continue

        X, Y = build_features(df)
        if len(X) == 0:
            continue

        X_all.append(X)
        Y_all.append(Y)

    X = np.vstack(X_all)
    Y = np.concatenate(Y_all)

    print("\n✅ Final training shape:", X.shape, Y.shape)

    # --------------------------------------
    # Scaling
    # --------------------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --------------------------------------
    # Train Random Forest
    # --------------------------------------
    print("\n🌲 Training RandomForest...")
    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=25,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_scaled, Y)

    # --------------------------------------
    # Save model + scaler
    # --------------------------------------
    joblib.dump(model, "rf_model.joblib")
    joblib.dump(scaler, "scaler_feat.joblib")

    print("\n🎉 Training complete!")
    print("Saved: rf_model.joblib, scaler_feat.joblib")
