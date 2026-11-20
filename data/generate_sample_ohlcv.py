"""
generate_sample_ohlcv.py — Generate realistic multi-symbol OHLCV dataset CSV for user testing & uploading.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Fix random seed for reproducible synthetic market structure
np.random.seed(42)

symbols = ["BTC/USDT", "ETH/USDT", "APPC/BTC", "STMX/USDT", "BNT/BTC"]
num_bars = 500  # 500 time-steps (e.g. 5-minute intervals)
start_time = pd.Timestamp("2021-06-14 00:00:00", tz="UTC")

base_prices = {
    "BTC/USDT": 40000.0,
    "ETH/USDT": 2500.0,
    "APPC/BTC": 0.00000250,
    "STMX/USDT": 0.035,
    "BNT/BTC": 0.0000850,
}

records = []

for symbol in symbols:
    price = base_prices[symbol]
    vol_base = 1000.0 if "USDT" in symbol else 50000.0
    
    for i in range(num_bars):
        ts = start_time + pd.Timedelta(minutes=5 * i)
        
        # Normal random walk return
        ret = np.random.normal(0.0001, 0.005)
        
        # Inject realistic volume surge / price spike anomalies (simulated pump setups)
        is_anomaly = False
        if i in [150, 320, 450]:  # Pump event locations
            ret = np.random.uniform(0.04, 0.08)  # 4% to 8% sudden price surge
            vol_mult = np.random.uniform(5.0, 10.0) # 5x to 10x volume spike
            is_anomaly = True
        else:
            vol_mult = np.random.lognormal(0, 0.3)
            
        open_price = price
        close_price = open_price * (1.0 + ret)
        high_price = max(open_price, close_price) * (1.0 + abs(np.random.normal(0.001, 0.002)))
        low_price = min(open_price, close_price) * (1.0 - abs(np.random.normal(0.001, 0.002)))
        volume = vol_base * vol_mult
        
        price = close_price  # Carry forward
        
        records.append({
            "timestamp": ts.isoformat(),
            "symbol": symbol,
            "open": round(open_price, 8),
            "high": round(high_price, 8),
            "low": round(low_price, 8),
            "close": round(close_price, 8),
            "volume": round(volume, 4),
        })

df = pd.DataFrame(records)

out_file = Path("sample_crypto_ohlcv.csv")
df.to_csv(out_file, index=False)

print(f"Generated sample OHLCV dataset ({len(df)} rows across {len(symbols)} symbols)")
print(f"File saved to: {out_file.resolve()}")
