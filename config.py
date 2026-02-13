"""
US Stock Screener — RS Rank + T-Value (Trend Quality)
All tuneable parameters live here.
"""
import os

from dotenv import load_dotenv

load_dotenv()

# ── Telegram ──────────────────────────────────────────────
TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")

# ── Schedule (UTC 22:00 = GMT+8 06:00) ───────────────────
CRON_HOUR: int = 22
CRON_MINUTE: int = 0
TIMEZONE: str = "Asia/Taipei"

# ── Stock Pool ────────────────────────────────────────────
MIN_PRICE: float = 10.01
MIN_AVG_DOLLAR_VOLUME_20D: float = 8_000_000
MIN_LISTING_DAYS: int = 252  # ≈ 1 year of trading days

# ── RS Rank (dual-period) ────────────────────────────────
RS20_DAYS: int = 20
RS60_DAYS: int = 60
RS20_PERCENTILE_CUTOFF: float = 85  # top 15 %
RS60_PERCENTILE_CUTOFF: float = 70  # top 30 %

# ── T-Value (Trend Quality) ──────────────────────────────
T_MA_PERIOD: int = 10
T_WINDOWS: dict[str, int] = {"T20": 20, "T60": 60, "T120": 120}
T60_PERCENTILE_CUTOFF: float = 70   # top 30 %
T_WEIGHTED_BLEND: float = 0.5       # blend ratio: std vs ATR-weighted
ATR_PERIOD: int = 20

# ── Auxiliary Filters ─────────────────────────────────────
VOL_20D_GT_50D: bool = True
VOLATILITY_EXCLUDE_TOP_PCT: float = 10
NEAR_52W_HIGH_PCT: float = 0.85

# ── Scoring ───────────────────────────────────────────────
SCORE_WEIGHTS: dict[str, float] = {
    "rs20": 0.35,
    "rs60": 0.15,
    "t_composite": 0.35,
    "vol_change": 0.15,
}

# ── Output ────────────────────────────────────────────────
TOP_N: int = 15
OUTPUT_DIR: str = "output"

# ── Data ──────────────────────────────────────────────────
DATA_DOWNLOAD_PERIOD: str = "15mo"
YFINANCE_BATCH_SIZE: int = 200       # smaller batches to avoid rate limit
YFINANCE_BATCH_DELAY: float = 2.0    # seconds between batches
YFINANCE_MAX_RETRIES: int = 3        # retry count per batch on rate limit
YFINANCE_RETRY_DELAY: float = 30.0   # base seconds to wait on 429
FORWARD_TRADING_DAYS: int = 63       # ≈ 3 months for backtest
