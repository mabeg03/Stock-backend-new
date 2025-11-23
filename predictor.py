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
    return 100 - (100 / (1 + rs))


def compute_macd(series):
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

        if len(closes) < 60:
            raise ValueError("Need at least 60 candles for prediction.")

        s = pd.Series(closes)
        window = s[-60:]

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

        last_volume = float(volumes[-1])

        return np.array([[
            lag1, ma7, ma21,
            ema12, ema26, ema50,
            std21, upper_bb, lower_bb,
            rsi14, macd_val, macd_sig_val, macd_hist_val,
            last_volume
        ]])

    def predict(self, ticker: str, days: int = 1):
        # Download recent data
        df = yf.download(ticker, period="1y", interval="1d", progress=False)
        if df.empty:
            raise ValueError("No stock data found")

        closes = df["Close"].values.reshape(-1) 
        volumes = df["Volume"].values

        X = self._build_feature_row(closes, volumes)
        X_scaled = self.scaler.transform(X)

        # Model predicts % return
        predicted_return = float(self.model.predict(X_scaled)[0])

        last_price = float(closes[-1])
        raw_pred_price = last_price * (1 + predicted_return)

        # Clamp to reasonable range: ±10%
        lower_bound = last_price * 0.90
        upper_bound = last_price * 1.10
        predicted_price = min(max(raw_pred_price, lower_bound), upper_bound)

        # --- Confidence based on recent volatility ---
        returns = pd.Series(closes).pct_change().tail(30)
        vol = float(returns.std()) if not returns.empty else 0.02  # default 2%

        # Map volatility to confidence (more volatile → lower confidence)
        # Rough mapping: vol ~ 1–5% → conf ~ 0.9–0.5
        confidence = 1 - (vol * 8)
        confidence = max(0.0, min(1.0, confidence))
        confidence = round(confidence, 3)

        return {
            "predicted": predicted_price,
            "confidence": confidence
        }

