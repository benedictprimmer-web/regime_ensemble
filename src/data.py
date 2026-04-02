"""
Polygon.io data fetcher with local CSV cache.

Requires POLYGON_API_KEY in .env (copy from .env.example).
Results are cached to data/cache/ — subsequent runs are instant.
"""

import os
from typing import Optional
import numpy as np
import pandas as pd
from pathlib import Path
from polygon import RESTClient
from dotenv import load_dotenv

load_dotenv()

CACHE_DIR = Path(__file__).parent.parent / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

_client: Optional[RESTClient] = None


def _get_client() -> RESTClient:
    global _client
    if _client is None:
        key = os.getenv("POLYGON_API_KEY")
        if not key:
            raise EnvironmentError(
                "POLYGON_API_KEY not set — copy .env.example to .env and add your key"
            )
        _client = RESTClient(api_key=key)
    return _client


def fetch_daily_bars(ticker: str, from_date: str, to_date: str) -> pd.DataFrame:
    """
    Fetch adjusted daily OHLCV bars for a ticker.

    Returns:
        DataFrame with DatetimeIndex and columns: open, high, low, close, volume
    """
    cache_path = CACHE_DIR / f"{ticker}_{from_date}_{to_date}.csv"
    if cache_path.exists():
        return pd.read_csv(cache_path, index_col="date", parse_dates=True)

    client = _get_client()
    bars = client.get_aggs(
        ticker,
        multiplier=1,
        timespan="day",
        from_=from_date,
        to=to_date,
        adjusted=True,
        sort="asc",
        limit=50000,
    )
    records = [
        {
            "date":   pd.to_datetime(b.timestamp, unit="ms"),
            "open":   b.open,
            "high":   b.high,
            "low":    b.low,
            "close":  b.close,
            "volume": b.volume,
        }
        for b in bars
    ]
    df = pd.DataFrame(records).set_index("date")
    df.to_csv(cache_path)
    return df


def log_returns(df: pd.DataFrame) -> pd.Series:
    """Daily log returns from a bars DataFrame."""
    return np.log(df["close"] / df["close"].shift(1)).dropna().rename("log_return")
