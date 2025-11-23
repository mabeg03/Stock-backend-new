import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
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
def build_features(df: pd.DataFrame):
    closes = df["Close"].values
    volumes = df["Volume"].values
    s = pd.Series(closes)

    X, y = [], []

    # use last 60 candles to predict next-day % return
    for i in range(60, len(closes)):
        window = s[i-60:i]
        last_close = closes[i-1]
        today_close = closes[i]

        # target: percentage return
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


if __name__ == "__main__":
    # Indian + US mix
    TICKERS = [
        # Indian
        "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
        "SBIN.NS", "KOTAKBANK.NS", "BHARTIARTL.NS", "ITC.NS", "ASIANPAINT.NS",
        # US
        "AAPL", "MSFT", "GOOGL", "AMZN", "META",
        "TSLA", "NVDA", "NFLX", "JPM", "V"
    ]

    X_all, y_all = [], []

    print("📥 Downloading data & building dataset...")
    for t in TICKERS:
        print(" →", t)
        df = yf.download(t, period="5y", interval="1d", progress=False)

        if df.empty or len(df) < 100:
            print("   Skipped (no/low data)")
            continue

        X, y = build_features(df)
        if len(X) == 0:
            print("   Skipped (no samples after feature build)")
            continue

        X_all.append(X)
        y_all.append(y)

    if not X_all:
        raise RuntimeError("No training data collected!")

    X = np.vstack(X_all)
    y = np.concatenate(y_all)

    print("✅ Final training shape:", X.shape, y.shape)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("🌲 Training XGBoost Regressor...")
    model = XGBRegressor(
        n_estimators=600,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="reg:squarederror",
        n_jobs=-1,
        random_state=42
    )

    model.fit(X_scaled, y)

    joblib.dump(model, "xgb_model.joblib")
    joblib.dump(scaler, "scaler_feat.joblib")

    print("🎉 Training complete! Saved xgb_model.joblib and scaler_feat.joblib")
