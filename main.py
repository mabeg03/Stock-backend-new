from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from predictor import HybridStockPredictor
import yfinance as yf
from datetime import datetime
import traceback

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = HybridStockPredictor()

def auto_fix_ticker(t):
    t = t.upper().strip()

    INDEX = {
        "NIFTY": "^NSEI",
        "BANKNIFTY": "^NSEBANK",
        "SENSEX": "^BSESN"
    }
    if t in INDEX:
        return INDEX[t]

    if "." in t:
        return t

    # NSE first
    if not yf.download(t + ".NS", period="1mo").empty:
        return t + ".NS"

    # BSE next
    if not yf.download(t + ".BO", period="1mo").empty:
        return t + ".BO"

    # Crypto
    if t in ["BTC", "ETH"]:
        return t + "-USD"

    return t


def get_live_price(ticker):
    """Return most accurate current market price available."""
    stock = yf.Ticker(ticker)

    # 1. Try fast_info
    try:
        p = stock.fast_info["last_price"]
        if p: return float(p)
    except:
        pass

    # 2. Try info
    try:
        info = stock.info
        for k in ["regularMarketPrice", "currentPrice", "open", "previousClose"]:
            if k in info and info[k] is not None:
                return float(info[k])
    except:
        pass

    # 3. Try 1m history
    try:
        df = stock.history(period="1d", interval="1m")
        if not df.empty:
            return float(df["Close"].iloc[-1])
    except:
        pass

    # 4. Try daily history
    try:
        df = stock.history(period="5d")
        if not df.empty:
            return float(df["Close"].iloc[-1])
    except:
        pass

    raise Exception("Unable to fetch live price")


@app.get("/predict")
async def predict_stock(ticker: str, days: int = 1):
    try:
        ticker = auto_fix_ticker(ticker)

        live_price = get_live_price(ticker)

        result = predictor.predict(ticker, days)

        # Historical prices for your UI
        hist = yf.Ticker(ticker).history(period="7d")
        history = [
            {"date": idx.strftime("%Y-%m-%d"), "price": float(row["Close"])}
            for idx, row in hist.iterrows()
        ]

        return {
            "ticker": ticker,
            "company_name": yf.Ticker(ticker).info.get("longName", ticker),
            "current_price": round(live_price, 2),
            "predicted_price": round(result["predicted"], 2),
            "confidence": result["confidence"],
            "days_ahead": days,
            "historical_prices": history,
            "timestamp": datetime.now().isoformat(),
            "currency_symbol": "₹" if ticker.endswith(".NS") or ticker.endswith(".BO") else "$"
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e))
