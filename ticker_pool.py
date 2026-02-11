"""
Fetch & clean the US common-stock universe.
Source: NASDAQ Trader symbol directory (official, free, updated daily).
"""
import io
import logging
import re

import pandas as pd
import requests

logger = logging.getLogger(__name__)

NASDAQ_TRADED_URL = (
    "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt"
)

EXCHANGE_MAP: dict[str, str] = {
    "Q": "NASDAQ",  # Global Select Market
    "G": "NASDAQ",  # Global Market
    "S": "NASDAQ",  # Capital Market
    "N": "NYSE",
    "A": "AMEX",    # NYSE American
    "P": "AMEX",    # NYSE Arca
}

_EXCLUDE_NAME = re.compile(
    r"ADR|ADS|American Depositary|Depositary Receipt|"
    r"Preferred|Warrant|Right[s ]|Unit[s ]|"
    r"Debenture|Note[s ]|Bond[s ]|"
    r"SPAC|Acquisition Corp|Blank Check|"
    r"ETN|Trust Units|%",
    re.IGNORECASE,
)

_EXCLUDE_TICKER = re.compile(r"\^|\.W$|\.U$|\.R$")


def fetch_ticker_pool() -> pd.DataFrame:
    """
    Download and filter the US equity universe.

    Returns
    -------
    DataFrame with columns: [symbol, exchange, yf_symbol]
        symbol   : canonical ticker  (e.g. AAPL, BRK.B)
        exchange : TradingView prefix (NASDAQ / NYSE / AMEX)
        yf_symbol: yfinance ticker    (e.g. AAPL, BRK-B)
    """
    logger.info("Downloading NASDAQ Trader symbol directory …")
    resp = requests.get(NASDAQ_TRADED_URL, timeout=60)
    resp.raise_for_status()

    lines = [
        ln for ln in resp.text.strip().split("\n")
        if not ln.startswith("File Creation Time")
    ]
    df = pd.read_csv(io.StringIO("\n".join(lines)), sep="|")

    # ── Filter rows ───────────────────────────────────────
    mask = (
        (df["Nasdaq Traded"] == "Y")
        & (df["Test Issue"] == "N")
        & (df["ETF"] == "N")
        & (df["Financial Status"] != "D")
        & df["Listing Exchange"].isin(EXCHANGE_MAP)
    )
    df = df.loc[mask].copy()
    df["exchange"] = df["Listing Exchange"].map(EXCHANGE_MAP)

    # Name-based exclusions (ADR, preferred, warrants …)
    df = df[~df["Security Name"].str.contains(_EXCLUDE_NAME, na=False)]

    # Symbol cleanup
    df["symbol"] = df["NASDAQ Symbol"].str.strip()
    df = df[~df["symbol"].str.contains(_EXCLUDE_TICKER, na=False)]

    # yfinance uses "-" for class shares; NASDAQ Trader uses "."
    df["yf_symbol"] = df["symbol"].str.replace(".", "-", regex=False)

    df = df[["symbol", "exchange", "yf_symbol"]].reset_index(drop=True)
    logger.info("Ticker pool: %d symbols", len(df))
    return df


def to_tv_ticker(yf_symbol: str, exchange: str) -> str:
    """
    Convert yfinance ticker → TradingView format.

    - Adds exchange prefix
    - Class-share dash → dot  (BRK-B → BRK.B)
      Only converts when suffix is a single letter to avoid
      mis-converting legitimate tickers with dashes.
    """
    tv_sym = re.sub(r"-([A-Z])$", r".\1", yf_symbol)
    return f"{exchange}:{tv_sym}"
