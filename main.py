from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
from predictor import HybridStockPredictor
from datetime import datetime

app = FastAPI()

# ---------------------------------------
# CORS
# ---------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = HybridStockPredictor()

# ---------------------------------------
# AUTO FIX TICKER
# ---------------------------------------
def auto_fix_ticker(t: str) -> str:
    t = t.upper().strip()

    INDEX = {"NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK", "SENSEX": "^BSESN"}
    if t in INDEX:
        return INDEX[t]

    # Direct test
    if not yf.download(t, period="1mo", progress=False).empty:
        return t

    # NSE
    if not yf.download(t + ".NS", period="1mo", progress=False).empty:
        return t + ".NS"

    # BSE
    if not yf.download(t + ".BO", period="1mo", progress=False).empty:
        return t + ".BO"

    raise HTTPException(status_code=404, detail=f"No valid stock found for: {t}")

# ---------------------------------------
# GET LIVE PRICE
# ---------------------------------------
def get_live_price(ticker: str):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}
    except:
        info = {}

    if "regularMarketPrice" in info:
        return info["regularMarketPrice"]

    df = stock.history(period="1d")
    if not df.empty:
        return df["Close"].iloc[-1]

    return None

# ---------------------------------------
# PREDICT ENDPOINT (WITH 7-DAY SUPPORT)
# ---------------------------------------
@app.get("/predict")
async def predict_stock(ticker: str, days: int = 1):
    ticker = auto_fix_ticker(ticker)

    live_price = get_live_price(ticker)
    if live_price is None:
        raise HTTPException(status_code=400, detail="Unable to fetch live price")

    # Call predictor (now returns 1-day + 7-day forecast)
    result = predictor.predict(ticker, days)

    # --------------------------
    # SAFE HISTORY FETCH
    # --------------------------
    history_df = yf.download(ticker, period="1mo", progress=False)
    history = []

    if not history_df.empty and "Close" in history_df:
        for i, p in zip(history_df.index, history_df["Close"]):
            try:
                history.append({"date": str(i), "price": float(p)})
            except:
                pass

    return {
        "ticker": ticker,
        "current_price": live_price,

        # NEW OUTPUT:
        "predicted_next_day": result.get("predicted_next_day"),
        "predicted_7_days": result.get("predicted_7_days", []),

        "confidence": result["confidence"],
        "days_ahead": days,
        "timestamp": datetime.now().isoformat(),
        "historical_prices": history
    }

# ---------------------------------------
# ROOT ROUTE
# ---------------------------------------
@app.get("/")
def home():
    return {"status": "Backend Running Successfully", "routes": ["/predict"]}
