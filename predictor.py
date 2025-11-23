import os
import numpy as np
import pandas as pd
import joblib
import yfinance as yf


# -------------------------------------------
# RSI Calculation
# -------------------------------------------
def compute_rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period).mean()
    ma_down = down.ewm(alpha=1/period).mean()
    rs = ma_up / (ma_down + 1e-9)
    return 100 - (100 / (1 + rs))


# -------------------------------------------
# MACD Calculation
# -------------------------------------------
def compute_macd(series):
    ema12 = series.ewm(span=12).mean()
    ema26 = series.ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    hist = macd - signal
    return macd, signal, hist


# -------------------------------------------
# Main Predictor Class
# -------------------------------------------
class HybridStockPredictor:
    def __init__(self):
        base_dir = os.path.dirname(__file__)
        self.model = joblib.load(os.path.join(base_dir, "rf_model.joblib"))
        self.scaler = joblib.load(os.path.join(base_dir, "scaler_feat.joblib"))

    # ----------------------------------------
    # Build feature set
    # ----------------------------------------
    def _build_feature_row(self, closes, volumes):
        closes = np.array(closes).reshape(-1)
        volumes = np.array(volumes).reshape(-1)

        if len(closes) < 60:
            raise ValueError("Minimum 60 closing prices required")

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

    # ----------------------------------------
    # Predict function with dynamic confidence
    # ----------------------------------------
    def predict(self, ticker: str, days: int = 1):
        try:
            df = yf.download(ticker, period="1y", progress=False)
        except:
            raise ValueError("Could not download stock data")

        if df.empty:
            raise ValueError(f"No market data for {ticker}")

        closes = df["Close"].values
        volumes = df["Volume"].values

        X = self._build_feature_row(closes, volumes)
        X_scaled = self.scaler.transform(X)

        # Final predicted % return
        predicted_return = float(self.model.predict(X_scaled)[0])
        last_price = float(closes[-1])
        predicted_price = last_price * (1 + predicted_return)

        # Sanity clamp
        predicted_price = max(predicted_price, last_price * 0.90)
        predicted_price = min(predicted_price, last_price * 1.10)

        # --------------------------------------
        # ★ Dynamic Confidence Calculation
        # --------------------------------------
        tree_predictions = np.array([
            tree.predict(X_scaled)[0]
            for tree in self.model.estimators_
        ])

        std_dev = np.std(tree_predictions)  # Uncertainty
        confidence = max(0.0, 1 - (std_dev * 12))  # Convert to 0–1 range
        confidence = min(1.0, confidence)
        confidence = round(confidence, 3)

        return {
            "predicted": predicted_price,
            "confidence": confidence
        }
