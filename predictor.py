    def predict(self, ticker: str, days: int = 1):
        # Download recent data
        df = yf.download(ticker, period="1y", interval="1d", progress=False)
        if df.empty:
            raise ValueError("No stock data found")

        # Ensure 1D arrays
        closes = df["Close"].values.reshape(-1)
        volumes = df["Volume"].values.reshape(-1)

        # Build feature row
        X = self._build_feature_row(closes, volumes)
        X_scaled = self.scaler.transform(X)

        # Model output = raw return signal (can be noisy / biased)
        raw_return = float(self.model.predict(X_scaled)[0])

        # --- Use recent trend to make it realistic ---
        returns_series = pd.Series(closes).pct_change()

        # last 10-day average trend
        if len(returns_series) > 10:
            trend_10 = float(returns_series.tail(10).mean())
        else:
            trend_10 = 0.0

        # last 30-day average trend
        if len(returns_series) > 30:
            trend_30 = float(returns_series.tail(30).mean())
        else:
            trend_30 = trend_10

        # combine model signal + recent trend
        # (60% model, 40% market trend)
        blended_return = (0.6 * raw_return) + (0.2 * trend_10) + (0.2 * trend_30)

        # Clamp to realistic daily move: -5% to +5%
        adjusted_return = max(-0.05, min(0.05, blended_return))

        last_price = float(closes[-1])
        predicted_price = last_price * (1 + adjusted_return)

        # Round for output
        predicted_price = round(predicted_price, 2)

        # --- Confidence based on volatility ---
        recent_vol_window = returns_series.tail(30)
        vol = float(recent_vol_window.std()) if not recent_vol_window.empty else 0.02

        # higher volatility → lower confidence
        # vol ~ 1–5% → conf ~ 0.9–0.5
        confidence = 1 - (vol * 8)
        confidence = max(0.0, min(1.0, confidence))
        confidence = round(confidence, 3)

        return {
            "predicted": predicted_price,
            "confidence": confidence
        }
