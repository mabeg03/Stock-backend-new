import os
import numpy as np
import pandas as pd
import joblib
import yfinance as yf


def compute_rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period).mean()
    ma_down = down.ewm(alpha=1/period).mean()
    rs = ma_up / (ma_down + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def compute_macd(series):
    ema12 = series.ewm(span=12).mean()
    ema26 = series.ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    hist = macd - signal
    return macd.fillna(0), signal.fillna(0), hist.fillna(0)


class HybridStockPredictor:
    def __init__(self):
        base_dir = os.path.dirname(__file__)
        self.model = joblib.load(os.path.join(base_dir, "xgb_model.joblib"))
        self.scaler = joblib.load(os.path.join(base_dir, "scaler_feat.joblib"))

    def _build_feature_row(self, closes, volumes):
        closes = np.array(closes).reshape(-1)
        volumes = np.array(volumes).reshape(-1)

        if len(closes) < 60:
            raise ValueError("Need at least 60 candles for prediction.")

        s = pd.Series(closes)
        w = s[-60:]

        lag1 = float(w.iloc[-1])
        ma7 = float(w.tail(7).mean())
        ma21 = float(w.tail(21).mean())
        ema12 = float(w.ewm(span=12).mean().iloc[-1])
        ema26 = float(w.ewm(span=26).mean().iloc[-1])
        ema50 = float(w.ewm(span=50).mean().iloc[-1])
        std21 = float(w.tail(21).std())

        upper_bb = ma21 + 2 * std21
        lower_bb = ma21 - 2 * std21

        rsi14 = float(compute_rsi(w).iloc[-1])

        macd_v, macd_s, macd_h = compute_macd(w)
        macd_val = float(macd_v.iloc[-1])
        macd_sig_val = float(macd_s.iloc[-1])
        macd_hist_val = float(macd_h.iloc[-1])

        last_volume = float(volumes[-1])

        features = np.array([
            lag1, ma7, ma21,
            ema12, ema26, ema50,
            std21, upper_bb, lower_bb,
            rsi14, macd_val, macd_sig_val, macd_hist_val,
            last_volume
        ], dtype=float)

        return features.reshape(1, -1)

    # ------------------------------------------------------------------------------------
    # ✅ FIXED predict() method (correctly inside class, no indent errors)
    # ------------------------------------------------------------------------------------
    def predict(self, ticker: str, days: int = 1):
        df = yf.download(ticker, period="1y", interval="1d", progress=False)
        if df.empty:
            return {"error": "No stock data found"}

        closes = df["Close"].values.reshape(-1)
        volumes = df["Volume"].values.reshape(-1)

        X = self._build_feature_row(closes, volumes)
        X_scaled = self.scaler.transform(X)

        raw_return = float(self.model.predict(X_scaled)[0])

        returns_series = pd.Series(closes).pct_change()

        trend_10 = float(returns_series.tail(10).mean()) if len(returns_series) > 10 else 0.0
        trend_30 = float(returns_series.tail(30).mean()) if len(returns_series) > 30 else trend_10

        blended_return = (0.6 * raw_return) + (0.2 * trend_10) + (0.2 * trend_30)

        adjusted_return = max(-0.05, min(0.05, blended_return))

        last_price = float(closes[-1])
        predicted_price = round(last_price * (1 + adjusted_return), 2)

        recent_vol = returns_series.tail(30)
        vol = float(recent_vol.std()) if not recent_vol.empty else 0.02

        confidence = max(0.0, min(1.0, 1 - (vol * 8)))
        confidence = round(confidence, 3)

        return {
            "predicted": predicted_price,
            "confidence": confidence
        }
