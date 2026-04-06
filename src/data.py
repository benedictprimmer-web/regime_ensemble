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
from io import StringIO
from urllib.request import urlopen
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


def vvix_levels(df: pd.DataFrame) -> pd.Series:
    """
    VVIX daily close levels from a bars DataFrame.

    VVIX is the volatility-of-volatility index (for VIX options). Like VIX,
    this is a level series and should not be treated as a return instrument.
    """
    return df["close"].rename("vvix")


def _read_cached_series(cache_path: Path, name: str) -> Optional[pd.Series]:
    """Read a cached Date-indexed CSV as a numeric series; return None if empty/invalid."""
    if not cache_path.exists():
        return None
    try:
        s = pd.read_csv(cache_path, index_col="Date", parse_dates=True).squeeze()
        if isinstance(s, pd.DataFrame):
            if s.shape[1] == 0:
                return None
            s = s.iloc[:, 0]
        s = pd.to_numeric(s, errors="coerce").dropna()
        if len(s) == 0:
            return None
        s.name = name
        return s
    except Exception:
        return None


def _extract_yf_close(raw) -> pd.Series:
    """
    Extract close series from yfinance output robustly across column layouts.
    Handles both flat and MultiIndex columns.
    """
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        if close.shape[1] == 0:
            return pd.Series(dtype=float)
        close = close.iloc[:, 0]
    return pd.to_numeric(close, errors="coerce").dropna()


def _fetch_cboe_series(index_name: str, from_date: str, to_date: str, value_col: str) -> pd.Series:
    """
    Fetch daily index history from CBOE CSV endpoints.
    index_name examples: VIX, VVIX
    value_col examples: CLOSE, VVIX
    """
    url = f"https://cdn.cboe.com/api/global/us_indices/daily_prices/{index_name}_History.csv"
    with urlopen(url, timeout=20) as resp:
        txt = resp.read().decode("utf-8")
    df = pd.read_csv(StringIO(txt))
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    s = pd.to_numeric(df[value_col], errors="coerce")
    out = pd.Series(s.values, index=df["DATE"], name=index_name.lower()).dropna()
    out = out[(out.index >= pd.to_datetime(from_date)) & (out.index <= pd.to_datetime(to_date))]
    return out.sort_index()


def fetch_vix_yfinance(from_date: str, to_date: str) -> pd.Series:
    """
    Fetch VIX daily close levels from Yahoo Finance (^VIX) via yfinance.

    Use this when the Polygon plan does not include I:VIX. yfinance provides
    ^VIX back to 1990 at no cost. Results are cached to the same data/cache/
    directory as Polygon data.

    Returns:
        pd.Series of VIX close levels named "vix", indexed by date.
    """
    cache_path = CACHE_DIR / f"VIX_yf_{from_date}_{to_date}.csv"
    cached = _read_cached_series(cache_path, "vix")
    if cached is not None:
        return cached

    import yfinance as yf
    raw = yf.download("^VIX", start=from_date, end=to_date, progress=False)
    vix = _extract_yf_close(raw)
    if len(vix) == 0:
        # Fallback: official CBOE history.
        vix = _fetch_cboe_series("VIX", from_date, to_date, value_col="CLOSE")
    vix.index.name = "Date"
    vix.name = "vix"
    vix.to_csv(cache_path)
    return vix


def fetch_vvix_yfinance(from_date: str, to_date: str) -> pd.Series:
    """
    Fetch VVIX daily close levels from Yahoo Finance (^VVIX) via yfinance.

    Useful when Polygon index access does not include I:VVIX. Results are
    cached to data/cache/ similarly to other index fetch helpers.

    Returns:
        pd.Series of VVIX close levels named "vvix", indexed by date.
    """
    cache_path = CACHE_DIR / f"VVIX_yf_{from_date}_{to_date}.csv"
    cached = _read_cached_series(cache_path, "vvix")
    if cached is not None:
        return cached

    import yfinance as yf
    raw = yf.download("^VVIX", start=from_date, end=to_date, progress=False)
    vvix = _extract_yf_close(raw)
    if len(vvix) == 0:
        # Fallback: official CBOE history.
        vvix = _fetch_cboe_series("VVIX", from_date, to_date, value_col="VVIX")
    vvix.index.name = "Date"
    vvix.name = "vvix"
    vvix.to_csv(cache_path)
    return vvix
