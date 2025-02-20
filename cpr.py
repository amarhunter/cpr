def fetch_historical_data(symbol_token):
    
    params = {
        "exchange": "NSE",
        "symboltoken": symbol_token,
        "interval": "ONE_MINUTE",
        "fromdate": "2025-01-08 09:00",
        "todate": "2025-01-25 09:16"
    }
    response = obj.getCandleData(params)
    df = pd.DataFrame(response["data"], columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)
    return df

def calculate_cpr(df):
    df['Pivot'] = (df['high'] + df['low'] + df['close']) / 3
    df['BC'] = (df['high'] + df['low']) / 2
    df['TC'] = (df['Pivot'] * 2) - df['BC']
    return df

def cpr_trading_strategy(symbol_token, exchange="NSE"):
    df = fetch_historical_data(symbol_token)
    if df.empty:
        print("No data available for symbol token", symbol_token)
        return
    df = calculate_cpr(df)

    if df.empty or len(df) == 0:
        print("Insufficient data for CPR calculation")
        return

    last_candle = df.iloc[-1] if not df.empty else None
    if last_candle is None:
        print("No valid last candle data")
        return

    price = last_candle['close']
    tc, bc = last_candle['TC'], last_candle['BC']

    if price > tc:
        print("Buy Signal: Price above TC:" + str(tc))
    elif price < bc:
        print("Sell Signal: Price below BC:" + str(bc))
    else:
        print("No Trade: Price within CPR range")


if __name__ == "__main__":
    SYMBOL_TOKEN = "7083"  # Example for NIFTY 50
    while True:
        cpr_trading_strategy(SYMBOL_TOKEN)
        time.sleep(10)  # Check every 5 minutes
