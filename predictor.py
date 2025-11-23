import os
import numpy as np
import pandas as pd
import joblib
import yfinance as yf


def compute_rsi(series, period=14):
    if len(series) < period + 1:
        return pd.Series([50] * len(series))

    delta = series.diff().fillna(0)
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period).mean()
    ma_down = down.ewm(alpha=1/period).mean() + 1e-9
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))


def compute_macd(series):
    if len(series) < 30:
        n = len(series)
        return (pd.Series([0]*n), pd.Series([0]*n), pd.Series([0]*n))

    ema12 = series.ewm(span=12).mean()
    ema26 = series.ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    hist = macd - signal
    return macd, signal, hist


class HybridStockPredictor:
    def __init__(self):
        base_dir = os.path.dirname(__file__)
        self.model = joblib.load(os.path.join(base_dir, "xgb_model.joblib"))
        self.scaler = joblib.load(os.path.join(base_dir, "scaler_feat.joblib"))

    def _build_feature_row(self, closes, volumes):
        closes = np.array(closes).reshape(-1)
        volumes = np.array(volumes).reshape(-1)

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
            float(window.tail(21).mean() + 2*window.tail(21).std()),  # Upper BB
            float(window.tail(21).mean() - 2*window.tail(21).std()),  # Lower BB
            rsi14,
            float(macd_v.iloc[-1]),
            float(macd_s.iloc[-1]),
            float(macd_h.iloc[-1]),
            float(volumes[-1]),
        ]

        return np.array([features])

    # ---------------- 1-DAY PREDICTION ----------------
    def predict_one(self, closes, volumes):
        X = self._build_feature_row(closes, volumes)
        X_scaled = self.scaler.transform(X)
        pred = float(self.model.predict(X_scaled)[0])
        return pred

    # ---------------- 7-DAY PREDICTION ----------------
    def predict_7day(self, closes, volumes):
        closes = list(closes)
        volumes = list(volumes)
        preds = []

        for i in range(7):
            next_price = self.predict_one(closes, volumes)
            preds.append(next_price)

            # append prediction as new close
            closes.append(next_price)

            # fake next-day volume = last volume (cannot predict volume)
            volumes.append(volumes[-1])

            # keep only last 60 for stability
            closes = closes[-60:]
            volumes = volumes[-60:]

        return preds

    # ---------------- MAIN API CALL ----------------
    def predict(self, ticker: str, days: int = 1):
        t = ticker.upper().strip()

        # Try NSE, raw, BSE
        df = (yf.download(t + ".NS", period="1y", interval="1d", progress=False) or
              yf.download(t, period="1y", interval="1d", progress=False) or
              yf.download(t + ".BO", period="1y", interval="1d", progress=False))

        if df.empty:
            return {"error": "No data found"}

        closes = df["Close"].values.reshape(-1)
        volumes = df["Volume"].values.reshape(-1)

        next_price = self.predict_one(closes, volumes)
        predictions_7d = self.predict_7day(closes, volumes)

        # --- Confidence ---
        returns = pd.Series(closes).pct_change().tail(30)
        vol = float(returns.std()) if not returns.empty else 0.02
        confidence = max(0, min(1, round(1 - vol * 8, 3)))

        return {
            "predicted_next_day": round(next_price, 2),
            "predicted_7_days": [
                {"day": i+1, "price": round(p, 2)} for i, p in enumerate(predictions_7d)
            ],
            "confidence": confidence
        }
