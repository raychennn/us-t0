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
    # ── ADR / Depositary ──────────────────────────────────
    r"ADR|ADS|American Depositary|Depositary Receipt|"
    # ── Preferred / Debt / Convertible ────────────────────
    r"Preferred|Warrant|Right[s ]|Unit[s ]|"
    r"Debenture|Note[s ]|Bond[s ]|Convertible|"
    # ── SPAC / Shell ──────────────────────────────────────
    r"SPAC|Acquisition Corp|Blank Check|"
    # ── ETN / ETP ─────────────────────────────────────────
    r"ETN|ETP|"
    # ── Funds / Trusts / CEFs ─────────────────────────────
    r"Trust Units|"
    r"\bFund\b|\bFunds\b|"
    r"Closed.?End|Close[d]?.?Ended|"
    r"\bCEF\b|Capital Allocation|"
    r"Municipal|Fixed.?Income|High.?Yield|Income Fund|"
    r"Investment Trust|Investors Trust|"
    r"Royalty Trust|Royalty Corp|"
    r"Gold Trust|Silver Trust|Precious Metal|"
    r"Physical Gold|Physical Silver|Physical Platinum|"
    r"Bullion|"
    # ── LP / MLP / Partnership ────────────────────────────
    r"Limited Partnership|Master Limited|\bL\.?P\.?\b|"
    r"\bMLP\b|"
    # ── REIT identifiers (optional — uncomment to exclude)─
    # r"Real Estate Investment Trust|\bREIT\b|"
    # ── Other non-operating structures ────────────────────
    r"Holding[s]? of Beneficial Interest|"
    r"Contingent Value Right|"
    r"%",  # fixed-rate preferred stock
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


# ─────────────────────────────────────────────────────────
#  Post-filter: yfinance quoteType verification
# ─────────────────────────────────────────────────────────

# Only these quoteTypes are considered common stocks
_ALLOWED_QUOTE_TYPES = {"EQUITY"}

# Additional name keywords checked at verification stage
# (catches things that slipped through the initial regex)
_VERIFY_EXCLUDE_NAME = re.compile(
    r"\bETF\b|\bFund\b|\bTrust\b|\bIndex\b|"
    r"\bFutures\b|\bOptions\b|\bSwap\b|"
    r"\bCommodity\b|\bCommodities\b|"
    r"\bGold\b.*\bTrust\b|\bSilver\b.*\bTrust\b|"
    r"Closed.?End|"
    r"\bLP\b|\bL\.P\.\b",
    re.IGNORECASE,
)


def verify_common_stocks(yf_symbols: list[str]) -> set[str]:
    """
    Check a small batch of tickers via yfinance to confirm they are
    common stocks (quoteType == 'EQUITY' and not a fund/trust by name).

    Designed to run on the final ~20-30 candidates, not the full universe.

    Returns the set of symbols that pass verification.
    """
    if not yf_symbols:
        return set()

    import yfinance as yf

    verified: set[str] = set()

    # Use Tickers object for batch info retrieval
    tickers = yf.Tickers(" ".join(yf_symbols))

    for sym in yf_symbols:
        try:
            info = tickers.tickers[sym].info
        except (KeyError, AttributeError) as exc:
            logger.debug("verify_common_stocks: %s info unavailable: %s",
                         sym, exc)
            # If we can't verify, give benefit of the doubt
            verified.add(sym)
            continue

        # Check quoteType
        qt = info.get("quoteType", "EQUITY")
        if qt not in _ALLOWED_QUOTE_TYPES:
            long_name = info.get("longName", sym)
            logger.info("Excluded %s: quoteType=%s (%s)", sym, qt, long_name)
            continue

        # Secondary name check (catches edge cases)
        long_name = info.get("longName", "") or ""
        short_name = info.get("shortName", "") or ""
        combined_name = f"{long_name} {short_name}"
        if _VERIFY_EXCLUDE_NAME.search(combined_name):
            logger.info("Excluded %s: name match (%s)", sym, long_name)
            continue

        verified.add(sym)

    logger.info(
        "quoteType verification: %d/%d passed",
        len(verified), len(yf_symbols),
    )
    return verified
