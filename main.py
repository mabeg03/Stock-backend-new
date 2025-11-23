from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
from predictor import HybridStockPredictor
from datetime import datetime

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = HybridStockPredictor()

# --------------------------
# SAFE TICKER VALIDATION
# --------------------------
def auto_fix_ticker(t: str) -> str:
    t = t.upper().strip()

    # Indexes
    INDEX = {"NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK", "SENSEX": "^BSESN"}
    if t in INDEX:
        return INDEX[t]

    # Direct validity check
    try:
        df = yf.download(t, period="1mo", progress=False)
        if not df.empty:
            return t
    except:
        pass

    # NSE
    try:
        df = yf.download(t + ".NS", period="1mo", progress=False)
        if not df.empty:
            return t + ".NS"
    except:
        pass

    # BSE
    try:
        df = yf.download(t + ".BO", period="1mo", progress=False)
        if not df.empty:
            return t + ".BO"
    except:
        pass

    raise HTTPException(
        status_code=404,
        detail=f"No valid stock found for: {t}"
    )

# --------------------------
# GET LIVE PRICE
# --------------------------
def get_live_price(ticker: str):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}
    except:
        info = {}

    price = None
    if "regularMarketPrice" in info:
        price = info["regularMarketPrice"]
    else:
        df = stock.history(period="1d")
        if not df.empty:
            price = df["Close"].iloc[-1]

    return price

# --------------------------
# PREDICT ENDPOINT
# --------------------------
@app.get("/predict")
async def predict_stock(ticker: str, days: int = 1):
    ticker = auto_fix_ticker(ticker)

    # Get live price safely
    live_price = get_live_price(ticker)
    if live_price is None:
        raise HTTPException(status_code=400, detail="Unable to fetch live price")

    # Predict
    result = predictor.predict(ticker, days)
    # --------------------------
    # SAFE HISTORY DATA
    # --------------------------
    history_df = yf.download(ticker, period="1mo", progress=False)

    history = []
    if history_df is not None and not history_df.empty and "Close" in history_df:
        for i, p in zip(history_df.index, history_df["Close"]):
            try:
                price = float(p)
                history.append({"date": str(i), "price": price})
            except:
                pass

    # If no history available → return empty list instead of crashing
    if not history:
        history = []



    return {
    "ticker": ticker,
    "predicted_price": result["predicted"],
    "current_price": live_price,
    "confidence": result["confidence"],
    "days_ahead": days,
    "timestamp": datetime.now().isoformat(),
    "historical_prices": history
    }


# --------------------------
# ROOT ROUTE (IMPORTANT!)
# --------------------------
@app.get("/")
def home():
    return {"status": "Backend Running Successfully", "routes": ["/predict"]}



