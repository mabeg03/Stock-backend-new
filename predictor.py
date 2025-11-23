import os
import numpy as np
import pandas as pd
import joblib
import yfinance as yf


# ------------------------- Indicators -------------------------
def compute_rsi(series, period=14):
    if len(series) < period + 1:
        return pd.Series([50] * len(series))

    delta = series.diff().fillna(0)
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period).mean()
    ma_down = ma_down = down.ewm(alpha=1/period).mean() + 1e-9

    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))


def compute_macd(series):
    if len(series) < 30:
        n = len(series)
        return (
            pd.Series([0] * n),
            pd.Series([0] * n),
            pd.Series([0] * n),
        )

    ema12 = series.ewm(span=12).mean()
    ema26 = series.ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    hist = macd - signal

    return macd, signal, hist


# ------------------------- Predictor Class -------------------------
class HybridStockPredictor:
    def __init__(self):
        base_dir = os.path.dirname(__file__)
        self.model = joblib.load(os.path.join(base_dir, "xgb_model.joblib"))
        self.scaler = joblib.load(os.path.join(base_dir, "scaler_feat.joblib"))

    # Build feature row for model
    def _build_feature_row(self, closes, volumes):
        closes = np.array(closes).reshape(-1)
        volumes = np.array(volumes).reshape(-1)

        if len(closes) < 60:
            raise ValueError("Need at least 60 candles for prediction.")

        s = pd.Series(closes)
        window = s[-60:]

        rsi14 = float(compute_rsi(window).iloc[-1])
        macd_v, macd_s, macd_h = compute_macd(window)

        features = [
            float(window.iloc[-1]),                       # LAG-1
            float(window.tail(7).mean()),                 # MA7
            float(window.tail(21).mean()),                # MA21
            float(window.ewm(span=12).mean().iloc[-1]),   # EMA12
            float(window.ewm(span=26).mean().iloc[-1]),   # EMA26
            float(window.ewm(span=50).mean().iloc[-1]),   # EMA50
            float(window.tail(21).std()),                 # STD21
            float(window.tail(21).mean() + 2 * window.tail(21).std()),  # Upper BB
            float(window.tail(21).mean() - 2 * window.tail(21).std()),  # Lower BB
            rsi14,                                        # RSI
            float(macd_v.iloc[-1]),                       # MACD
            float(macd_s.iloc[-1]),                       # MACD Signal
            float(macd_h.iloc[-1]),                       # MACD Histogram
            float(volumes[-1]),                           # Last volume
        ]

        return np.array([features])

    # ------------------------- PREDICT -------------------------
    def predict(self, ticker: str, days: int = 1):
        t = ticker.upper().strip()

        # Try NSE first
        df = yf.download(t + ".NS", period="1y", interval="1d", progress=False)

        # Try raw ticker
        if df.empty:
            df = yf.download(t, period="1y", interval="1d", progress=False)

        # Try BSE
        if df.empty:
            df = yf.download(t + ".BO", period="1y", interval="1d", progress=False)

        if df.empty:
            return {"error": f"No market data found for {ticker}"}

        # 1D arrays
        closes = df["Close"].values.reshape(-1)
        volumes = df["Volume"].values.reshape(-1)

        # Build features
        X = self._build_feature_row(closes, volumes)
        X_scaled = self.scaler.transform(X)

        # ---------------- NEW: Model predicts next-day PRICE ----------------
        raw_price = float(self.model.predict(X_scaled)[0])

        last_price = float(closes[-1])

        # Safety clamp (±15%)
        lower = last_price * 0.85
        upper = last_price * 1.15
        predicted_price = min(max(raw_price, lower), upper)
        predicted_price = round(predicted_price, 2)

        # ---------------- Confidence ----------------
        returns = pd.Series(closes).pct_change().tail(30)
        vol = float(returns.std()) if not returns.empty else 0.02

        confidence = 1 - (vol * 8)       # higher volatility => lower confidence
        confidence = max(0.0, min(1.0, confidence))
        confidence = round(confidence, 3)

        return {
            "predicted": predicted_price,
            "confidence": confidence
        }
