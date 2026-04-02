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


def _safe_cache_name(ticker: str, from_date: str, to_date: str) -> str:
    """Return a filesystem-safe cache filename (replaces ':' for index tickers like I:VIX)."""
    safe_ticker = ticker.replace(":", "_")
    return f"{safe_ticker}_{from_date}_{to_date}.csv"


def fetch_daily_bars(ticker: str, from_date: str, to_date: str) -> pd.DataFrame:
    """
    Fetch adjusted daily OHLCV bars for a ticker.

    Supports equity tickers (SPY, QQQ, IWM) and Polygon index tickers
    (e.g. I:VIX). For indices, volume will be zero/null — this is expected.

    Returns:
        DataFrame with DatetimeIndex and columns: open, high, low, close, volume
    """
    cache_path = CACHE_DIR / _safe_cache_name(ticker, from_date, to_date)
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


def fetch_multi(tickers: list, from_date: str, to_date: str) -> dict:
    """
    Fetch daily bars for multiple tickers.

    Returns:
        dict mapping ticker → DataFrame (same format as fetch_daily_bars)
    """
    return {t: fetch_daily_bars(t, from_date, to_date) for t in tickers}


def vix_levels(df: pd.DataFrame) -> pd.Series:
    """
    VIX daily close levels from a bars DataFrame.

    VIX is a level, not a return instrument — use this instead of log_returns()
    when working with I:VIX data.
    """
    return df["close"].rename("vix")
