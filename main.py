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

    # Index handling
    INDEX = {"NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK", "SENSEX": "^BSESN"}
    if t in INDEX:
        return INDEX[t]

    # If user already gave correct symbol
    try:
        df = yf.download(t, period="1mo", progress=False)
        if not df.empty:
            return t
    except:
        pass

    # Try NSE
    try:
        df = yf.download(t + ".NS", period="1mo", progress=False)
        if not df.empty:
            return t + ".NS"
    except:
        pass

    # Try BSE
    try:
        df = yf.download(t + ".BO", period="1mo", progress=False)
        if not df.empty:
            return t + ".BO"
    except:
        pass

    # Crypto
    if t in ["BTC", "ETH"]:
        return t + "-USD"

    # Nothing worked
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

    return {
        "ticker": ticker,
        "predicted_price": result["predicted"],
        "current_price": live_price,
        "confidence": result["confidence"],
        "days_ahead": days,
        "timestamp": datetime.now().isoformat()
    }

# --------------------------
# ROOT ROUTE (IMPORTANT!)
# --------------------------
@app.get("/")
def home():
    return {"status": "Backend Running Successfully", "routes": ["/predict"]}
