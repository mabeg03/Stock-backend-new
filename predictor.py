import os
import numpy as np
import pandas as pd
import yfinance as yf
import joblib

# ------------------------
# Feature helpers (copied from train_rf_model.py)
# ------------------------

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

def make_single_feature_row(closes, volumes):
    """
    Build ONE feature row (14 features) from the last 60 candles.
    This matches the training logic in train_rf_model.py.
    """
    closes = np.array(closes).reshape(-1)
    volumes = np.array(volumes).reshape(-1)

    if len(closes) < 60:
        raise ValueError("Not enough data to build features (need at least 60 candles).")

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

    features = np.array([[
        lag1, ma7, ma21,
        ema12, ema26, ema50,
        std21, upper_bb, lower_bb,
        rsi14, macd_val, macd_sig_val, macd_hist_val,
        last_volume
    ]])

    return features


# ------------------------
# Main predictor class
# ------------------------

class HybridStockPredictor:
    def __init__(self):
        base_dir = os.path.dirname(__file__)
        model_path = os.path.join(base_dir, "models", "rf_model.joblib")
        scaler_path = os.path.join(base_dir, "models", "scaler_feat.joblib")

        if not os.path.exists(model_path):
            raise RuntimeError(f"Model file not found: {model_path}")
        if not os.path.exists(scaler_path):
            raise RuntimeError(f"Scaler file not found: {scaler_path}")

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

    def predict(self, ticker: str, days: int = 1):
        """
        Returns a dict with keys: 'predicted' and 'confidence'
        Compatible with main.py
        """
        # Get historical data
        df = yf.download(ticker, period="1y", progress=False)
        if df.empty:
            raise ValueError("No historical data available for this ticker.")

        closes = df["Close"].astype(float).values
        volumes = df["Volume"].astype(float).values

        # Build features from last 60 candles
        X_feat = make_single_feature_row(closes, volumes)   # shape (1, 14)
        X_scaled = self.scaler.transform(X_feat)

        # Predict next-day price using RF
        pred_price = float(self.model.predict(X_scaled)[0])

        # Simple dummy confidence: based on tree variance (optional improvement later)
        # For now, just return a reasonable fixed value:
        confidence = 0.80

        # If user asks many days ahead, reuse same prediction (you can improve later)
        return {
            "predicted": pred_price,
            "confidence": confidence
        }
