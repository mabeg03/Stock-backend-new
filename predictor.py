import os
import numpy as np
import pandas as pd
import joblib
import yfinance as yf


# ======================================================
# TECHNICAL INDICATORS
# ======================================================
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


# ======================================================
# SAFE TICKER DOWNLOAD
# ======================================================
def safe_download(ticker):
    """Try exact ticker, NSE, then BSE — safely."""
    tests = [ticker, ticker + ".NS", ticker + ".BO"]

    for tk in tests:
        try:
            df = yf.download(tk, period="1y", interval="1d", progress=False)
            if df is not None and not df.empty:
                return df
        except:
            pass

    raise ValueError(f"❌ No valid price data found for: {ticker}")


# ======================================================
# PREDICTOR CLASS
# ======================================================
class HybridStockPredictor:
    def __init__(self):
        base_dir = os.path.dirname(__file__)
        self.model = joblib.load(os.path.join(base_dir, "xgb_model.joblib"))
        self.scaler = joblib.load(os.path.join(base_dir, "scaler_feat.joblib"))

    # ------------------------------------------------------
    # BUILD FEATURES
    # ------------------------------------------------------
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
        macd_sig = float(macd_s.iloc[-1])
        macd_hist = float(macd_h.iloc[-1])

        last_volume = float(volumes[-1])

        return np.array([[lag1, ma7, ma21,
                          ema12, ema26, ema50,
                          std21, upper_bb, lower_bb,
                          rsi14, macd_val, macd_sig, macd_hist,
                          last_volume]])

    # ------------------------------------------------------
    # PREDICT FOR 1–7 DAYS
    # ------------------------------------------------------
    def predict(self, ticker: str, days: int = 1):
    # ---- STEP 1: SAFE DATA DOWNLOAD ----
    df = yf.download(ticker, period="1y", interval="1d", progress=False)
    if df is None or df.empty or "Close" not in df:
        return {"predicted": None, "confidence": 0}

    closes = df["Close"].values.reshape(-1)
    volumes = df["Volume"].values.reshape(-1)

    # ---- STEP 2: NEED MINIMUM 60 CANDLES ----
    if len(closes) < 60:
        return {"predicted": None, "confidence": 0}

    try:
        # ---- BUILD FEATURES ----
        X = self._build_feature_row(closes, volumes)
        X_scaled = self.scaler.transform(X)

        # ---- MODEL RETURN ----
        raw_return = float(self.model.predict(X_scaled)[0])

    except Exception as e:
        print("Model error:", e)
        return {"predicted": None, "confidence": 0}

    # ---- RECENT TREND ----
    returns_series = pd.Series(closes).pct_change()

    trend_10 = returns_series.tail(10).mean() if len(returns_series) >= 10 else 0
    trend_30 = returns_series.tail(30).mean() if len(returns_series) >= 30 else trend_10

    # ---- BLEND SIGNAL + TREND ----
    blended_return = (0.6 * raw_return) + (0.2 * trend_10) + (0.2 * trend_30)

    # ---- REALISTIC LIMIT ----
    adjusted_return = max(-0.05, min(0.05, blended_return))

    last_price = float(closes[-1])
    predicted_price = round(last_price * (1 + adjusted_return), 2)

    # ---- CONFIDENCE ----
    vol = returns_series.tail(30).std() if len(returns_series) > 30 else 0.02
    confidence = round(max(0.0, min(1.0, 1 - (vol * 8))), 3)

    return {
        "predicted": predicted_price,
        "confidence": confidence
    }
