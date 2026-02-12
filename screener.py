"""
Screening Pipeline
══════════════════
Pool → Download → Filter → RS(20/60) → T-Value → Aux → Score
     → quoteType Verify → Top N

Supports:
  • Live screening   (as_of_date=None → uses latest market data)
  • Back-testing     (as_of_date=date → screens as of that day,
                      then evaluates 3-month forward performance)
  • Symbol diagnosis (show pass/fail for every filter step)
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytz
import yfinance as yf

import config
from indicators import (
    calc_hist_volatility,
    calc_rs_return,
    calc_t_value,
    calc_volume_ratio,
    pct_from_52w_high,
    rank_percentile,
)
from ticker_pool import fetch_ticker_pool, to_tv_ticker, verify_common_stocks

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
#  Result Container
# ─────────────────────────────────────────────────────────

@dataclass
class ScreenResult:
    """Holds everything produced by a single screening run."""
    top: pd.DataFrame = field(default_factory=pd.DataFrame)
    txt_path: str = ""
    as_of: date | None = None
    # intermediate data kept for diagnose / forward-perf
    pool_data: dict[str, pd.DataFrame] = field(default_factory=dict)
    sym_exchange: dict[str, str] = field(default_factory=dict)
    rs20_ranks: pd.Series = field(default_factory=pd.Series)
    rs60_ranks: pd.Series = field(default_factory=pd.Series)
    t_values: pd.DataFrame = field(default_factory=pd.DataFrame)


# ─────────────────────────────────────────────────────────
#  Data Download
# ─────────────────────────────────────────────────────────

def _download_prices(
    symbols: list[str],
    as_of_date: date | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Batch-download OHLCV from Yahoo Finance with rate-limit handling.

    When *as_of_date* is given, fetches a date range that covers both
    the look-back for indicators and the 3-month forward window.
    """
    all_data: dict[str, pd.DataFrame] = {}
    bs = config.YFINANCE_BATCH_SIZE

    dl_kwargs: dict = dict(
        group_by="ticker", auto_adjust=True,
        threads=True, progress=False,
    )
    if as_of_date is not None:
        dl_kwargs["start"] = as_of_date - timedelta(days=500)
        dl_kwargs["end"] = min(
            as_of_date + timedelta(days=130),
            date.today() + timedelta(days=1),
        )
    else:
        dl_kwargs["period"] = config.DATA_DOWNLOAD_PERIOD

    total_batches = (len(symbols) + bs - 1) // bs

    for i in range(0, len(symbols), bs):
        batch = symbols[i: i + bs]
        batch_num = i // bs + 1

        # Retry loop for rate limiting
        raw = None
        for attempt in range(1, config.YFINANCE_MAX_RETRIES + 1):
            logger.info(
                "Downloading batch %d/%d (%d tickers)%s …",
                batch_num, total_batches, len(batch),
                f" attempt {attempt}" if attempt > 1 else "",
            )
            try:
                raw = yf.download(tickers=batch, **dl_kwargs)
                break  # success
            except Exception as exc:
                exc_str = str(exc).lower()
                if "rate" in exc_str or "429" in exc_str or "too many" in exc_str:
                    wait = config.YFINANCE_RETRY_DELAY * attempt
                    logger.warning(
                        "Batch %d rate-limited (attempt %d/%d), "
                        "waiting %.0fs …",
                        batch_num, attempt, config.YFINANCE_MAX_RETRIES, wait,
                    )
                    time.sleep(wait)
                else:
                    logger.warning(
                        "Batch %d download error: %s", batch_num, exc,
                    )
                    break  # non-rate-limit error, skip batch

        if raw is None or raw.empty:
            continue

        # Parse results
        cols = ["Open", "High", "Low", "Close", "Volume"]
        if len(batch) == 1:
            sym = batch[0]
            try:
                df = raw[cols].dropna()
                if len(df) >= config.MIN_LISTING_DAYS:
                    all_data[sym] = df
            except KeyError:
                pass
        else:
            for sym in batch:
                try:
                    df = raw[sym][cols].dropna()
                    if len(df) >= config.MIN_LISTING_DAYS:
                        all_data[sym] = df
                except (KeyError, TypeError):
                    continue

        # Delay between batches to stay under rate limits
        if batch_num < total_batches:
            time.sleep(config.YFINANCE_BATCH_DELAY)

    logger.info("Downloaded data for %d symbols", len(all_data))
    return all_data


# ─────────────────────────────────────────────────────────
#  Trim Data to as-of Date
# ─────────────────────────────────────────────────────────

def _trim_to_date(
    data: dict[str, pd.DataFrame], as_of_date: date | None,
) -> dict[str, pd.DataFrame]:
    """Keep only rows up to *as_of_date* for screening purposes."""
    if as_of_date is None:
        return data
    trimmed: dict[str, pd.DataFrame] = {}
    for sym, df in data.items():
        mask = df.index.date <= as_of_date
        t = df.loc[mask]
        if len(t) >= config.MIN_LISTING_DAYS:
            trimmed[sym] = t
    return trimmed


# ─────────────────────────────────────────────────────────
#  Pool Filters
# ─────────────────────────────────────────────────────────

def _apply_pool_filters(
    data: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    filtered: dict[str, pd.DataFrame] = {}
    for sym, df in data.items():
        if len(df) < config.MIN_LISTING_DAYS:
            continue
        last_close = float(df["Close"].iloc[-1])
        if last_close < config.MIN_PRICE:
            continue
        recent20 = df.tail(20)
        avg_dv = float((recent20["Close"] * recent20["Volume"]).mean())
        if avg_dv < config.MIN_AVG_DOLLAR_VOLUME_20D:
            continue
        filtered[sym] = df

    logger.info("After pool filters: %d symbols", len(filtered))
    return filtered


# ─────────────────────────────────────────────────────────
#  RS Rank (dual-period)
# ─────────────────────────────────────────────────────────

def _calc_all_rs(
    data: dict[str, pd.DataFrame],
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Returns (rs20_raw, rs60_raw, rs20_rank, rs60_rank).
    Each is a pd.Series indexed by symbol.
    """
    rs20_raw: dict[str, float] = {}
    rs60_raw: dict[str, float] = {}

    for sym, df in data.items():
        c = df["Close"].to_numpy(dtype=float)
        r20 = calc_rs_return(c, config.RS20_DAYS)
        r60 = calc_rs_return(c, config.RS60_DAYS)
        if r20 is not None:
            rs20_raw[sym] = r20
        if r60 is not None:
            rs60_raw[sym] = r60

    s20 = pd.Series(rs20_raw)
    s60 = pd.Series(rs60_raw)
    r20 = rank_percentile(s20)
    r60 = rank_percentile(s60)
    return s20, s60, r20, r60


def _filter_rs(
    rs20_rank: pd.Series, rs60_rank: pd.Series,
) -> set[str]:
    """
    Pass if:
      • RS(20) percentile ≥ 85
      • RS(60) percentile ≥ 70
      • RS(20) pct > RS(60) pct  (momentum acceleration)
    """
    common = rs20_rank.index.intersection(rs60_rank.index)
    r20 = rs20_rank.loc[common]
    r60 = rs60_rank.loc[common]

    mask = (
        (r20 >= config.RS20_PERCENTILE_CUTOFF)
        & (r60 >= config.RS60_PERCENTILE_CUTOFF)
        & (r20 > r60)
    )
    passed = set(mask[mask].index)
    logger.info(
        "RS filter: %d passed (RS20≥%s & RS60≥%s & accel)",
        len(passed), config.RS20_PERCENTILE_CUTOFF, config.RS60_PERCENTILE_CUTOFF,
    )
    return passed


# ─────────────────────────────────────────────────────────
#  T-Value Calculation
# ─────────────────────────────────────────────────────────

def _calc_all_t(
    symbols: set[str],
    data: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Compute T-values for a subset of symbols."""
    records: list[dict] = []
    for sym in symbols:
        df = data[sym]
        c = df["Close"].to_numpy(dtype=float)
        h = df["High"].to_numpy(dtype=float)
        lo = df["Low"].to_numpy(dtype=float)
        row: dict = {"symbol": sym}
        for label, window in config.T_WINDOWS.items():
            row[label] = calc_t_value(c, h, lo, window)
        records.append(row)
    return pd.DataFrame(records).set_index("symbol")


def _filter_t(t_df: pd.DataFrame) -> pd.DataFrame:
    """T60 ≥ 70th percentile & T120 ≥ median."""
    t_df = t_df.dropna(subset=["T60"])
    if t_df.empty:
        return t_df

    t60_cut = t_df["T60"].quantile(config.T60_PERCENTILE_CUTOFF / 100)
    passed = t_df[t_df["T60"] >= t60_cut].copy()

    if "T120" in passed.columns:
        t120_med = t_df["T120"].median()
        passed = passed[passed["T120"].fillna(-np.inf) >= t120_med]

    logger.info("After T-value filter: %d symbols", len(passed))
    return passed


# ─────────────────────────────────────────────────────────
#  Auxiliary Filters
# ─────────────────────────────────────────────────────────

def _apply_aux_filters(
    symbols: pd.Index,
    data: dict[str, pd.DataFrame],
) -> tuple[list[str], dict[str, dict]]:
    """
    Returns (passing_symbols, aux_metrics_per_symbol).
    """
    candidates: list[str] = []
    aux: dict[str, dict] = {}

    for sym in symbols:
        df = data[sym]
        c = df["Close"].to_numpy(dtype=float)
        v = df["Volume"].to_numpy(dtype=float)

        vr = calc_volume_ratio(v)
        if config.VOL_20D_GT_50D and (vr is None or vr < 1.0):
            continue

        p52 = pct_from_52w_high(c)
        if p52 is None or p52 < config.NEAR_52W_HIGH_PCT:
            continue

        hvol = calc_hist_volatility(c)
        candidates.append(sym)
        aux[sym] = {"vol_ratio": vr or 1.0, "pct_52w": p52, "hvol": hvol}

    # Exclude top-decile volatility
    if candidates:
        vols = pd.Series({s: aux[s]["hvol"] for s in candidates}).dropna()
        if not vols.empty:
            cut = vols.quantile(1 - config.VOLATILITY_EXCLUDE_TOP_PCT / 100)
            candidates = [s for s in candidates if (vols.get(s, 0) <= cut)]

    logger.info("After auxiliary filters: %d symbols", len(candidates))
    return candidates, aux


# ─────────────────────────────────────────────────────────
#  Scoring
# ─────────────────────────────────────────────────────────

def _score_and_rank(
    symbols: list[str],
    rs20_rank: pd.Series,
    rs60_rank: pd.Series,
    t_df: pd.DataFrame,
    aux: dict[str, dict],
    sym_exchange: dict[str, str],
) -> pd.DataFrame:
    rows: list[dict] = []
    for sym in symbols:
        rows.append({
            "symbol": sym,
            "rs20_pct": rs20_rank.get(sym, 0),
            "rs60_pct": rs60_rank.get(sym, 0),
            "T20": t_df.at[sym, "T20"] if sym in t_df.index else None,
            "T60": t_df.at[sym, "T60"] if sym in t_df.index else None,
            "T120": t_df.at[sym, "T120"] if sym in t_df.index else None,
            "vol_ratio": aux.get(sym, {}).get("vol_ratio"),
            "pct_52w": aux.get(sym, {}).get("pct_52w"),
        })

    result = pd.DataFrame(rows).set_index("symbol")
    if result.empty:
        return result

    # Percentile within final pool for blending
    rp20 = result["rs20_pct"].rank(pct=True)
    rp60 = result["rs60_pct"].rank(pct=True)
    tp = result["T60"].rank(pct=True)
    vp = result["vol_ratio"].rank(pct=True)

    w = config.SCORE_WEIGHTS
    result["score"] = (
        w["rs20"] * rp20
        + w["rs60"] * rp60
        + w["t_composite"] * tp
        + w["vol_change"] * vp
    )

    # Bonus for T20 > T60 (short-term acceleration)
    accel = result["T20"].fillna(0) > result["T60"].fillna(0)
    result.loc[accel, "score"] += 0.05

    result = result.sort_values("score", ascending=False)

    # Add exchange / TV ticker
    result["exchange"] = [sym_exchange.get(s, "NYSE") for s in result.index]
    result["tv_ticker"] = [
        to_tv_ticker(s, sym_exchange.get(s, "NYSE")) for s in result.index
    ]
    return result


# ─────────────────────────────────────────────────────────
#  TXT Output (TradingView watchlist)
# ─────────────────────────────────────────────────────────

def _write_txt(top: pd.DataFrame, date_str: str) -> str:
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    filename = f"US-RS+T_{date_str}.txt"
    path = os.path.join(config.OUTPUT_DIR, filename)
    lines = [row["tv_ticker"] for _, row in top.iterrows()]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    logger.info("Wrote %s (%d lines)", path, len(lines))
    return path


# ═════════════════════════════════════════════════════════
#  Main Entry Point
# ═════════════════════════════════════════════════════════

def run_screening(as_of_date: date | None = None) -> ScreenResult:
    """
    Execute the full pipeline.

    Parameters
    ----------
    as_of_date : date, optional
        If given, screen as of this historical date and download
        enough forward data for performance evaluation.
        If None, screen with the latest available data.
    """
    tz = pytz.timezone(config.TIMEZONE)
    effective_date = as_of_date or datetime.now(tz).date()
    date_str = effective_date.strftime("%y%m%d")
    result = ScreenResult(as_of=effective_date)

    # 0 — ticker pool
    pool = fetch_ticker_pool()
    result.sym_exchange = dict(zip(pool["yf_symbol"], pool["exchange"]))
    yf_syms = pool["yf_symbol"].tolist()

    # 1 — download
    full_data = _download_prices(yf_syms, as_of_date)
    result.pool_data = full_data

    # 2 — trim to as-of date for screening
    data = _trim_to_date(full_data, as_of_date)

    # 3 — pool filters
    data = _apply_pool_filters(data)

    # 4 — RS rank
    _, _, rs20_rank, rs60_rank = _calc_all_rs(data)
    result.rs20_ranks = rs20_rank
    result.rs60_ranks = rs60_rank
    rs_pass = _filter_rs(rs20_rank, rs60_rank)

    if not rs_pass:
        logger.warning("No symbols passed RS filter")
        return result

    # 5 — T-value
    t_all = _calc_all_t(rs_pass, data)
    result.t_values = t_all
    t_pass = _filter_t(t_all)

    if t_pass.empty:
        logger.warning("No symbols passed T-value filter")
        return result

    # 6 — auxiliary
    aux_syms, aux_data = _apply_aux_filters(t_pass.index, data)

    if not aux_syms:
        logger.warning("No symbols passed auxiliary filters")
        return result

    # 7 — score
    top_all = _score_and_rank(
        aux_syms, rs20_rank, rs60_rank, t_pass, aux_data, result.sym_exchange,
    )

    # 8 — verify top candidates are common stocks (not funds/trusts)
    # Check 3× TOP_N to have buffer after exclusions
    verify_n = min(len(top_all), config.TOP_N * 3)
    candidates = top_all.head(verify_n).index.tolist()
    verified = verify_common_stocks(candidates)
    top_all = top_all[top_all.index.isin(verified)]

    top = top_all.head(config.TOP_N)
    result.top = top

    # 9 — TXT
    result.txt_path = _write_txt(top, date_str)

    return result


# ═════════════════════════════════════════════════════════
#  Forward Performance (for back-testing)
# ═════════════════════════════════════════════════════════

def calc_forward_performance(
    sr: ScreenResult,
) -> pd.DataFrame | None:
    """
    For each symbol in ``sr.top``, compute 3-month forward metrics
    using data stored in ``sr.pool_data``.

    Returns DataFrame with: max_price, min_price, max_ret%, mdd%
    """
    if sr.top.empty or sr.as_of is None:
        return None

    rows: list[dict] = []
    for sym in sr.top.index:
        if sym not in sr.pool_data:
            continue
        df = sr.pool_data[sym]
        fwd = df[df.index.date > sr.as_of].head(config.FORWARD_TRADING_DAYS)
        if fwd.empty:
            rows.append({"symbol": sym, "fwd_days": 0})
            continue

        entry = float(df[df.index.date <= sr.as_of]["Close"].iloc[-1])
        closes = fwd["Close"].to_numpy(dtype=float)

        max_p = float(np.max(closes))
        min_p = float(np.min(closes))
        max_ret = (max_p / entry - 1) * 100

        # MDD from running peak (starting from entry price)
        running_peak = entry
        max_dd = 0.0
        for p in closes:
            if p > running_peak:
                running_peak = p
            dd = (running_peak - p) / running_peak * 100
            if dd > max_dd:
                max_dd = dd

        rows.append({
            "symbol": sym,
            "entry": round(entry, 2),
            "max_price": round(max_p, 2),
            "min_price": round(min_p, 2),
            "max_ret_pct": round(max_ret, 1),
            "mdd_pct": round(max_dd, 1),
            "fwd_days": len(closes),
        })

    if not rows:
        return None
    return pd.DataFrame(rows).set_index("symbol")


# ═════════════════════════════════════════════════════════
#  Symbol Diagnosis
# ═════════════════════════════════════════════════════════

def diagnose_symbol(
    symbol: str,
    sr: ScreenResult,
) -> list[tuple[bool | None, str]]:
    """
    Replay every filter step for *symbol*, returning a checklist:
        [(passed: bool|None, description: str), …]
    None means "could not evaluate" (missing data).

    Requires ``run_screening()`` to have been called first so that
    RS ranks and T-values are populated in *sr*.
    """
    checks: list[tuple[bool | None, str]] = []
    yf_sym = symbol.replace(".", "-")   # allow TV-style input

    # -- pool data --
    trimmed = _trim_to_date(sr.pool_data, sr.as_of)
    if yf_sym not in trimmed:
        checks.append((False, f"不在資料池中（無歷史數據或不足 {config.MIN_LISTING_DAYS} 天）"))
        return checks

    df = trimmed[yf_sym]
    c = df["Close"].to_numpy(dtype=float)
    v = df["Volume"].to_numpy(dtype=float)
    h = df["High"].to_numpy(dtype=float)
    lo = df["Low"].to_numpy(dtype=float)
    last_close = float(c[-1])

    # -- price --
    ok = last_close >= config.MIN_PRICE
    checks.append((ok, f"股價 ${last_close:.2f} {'≥' if ok else '<'} ${config.MIN_PRICE}"))

    # -- dollar volume --
    dv = float((df.tail(20)["Close"] * df.tail(20)["Volume"]).mean())
    ok = dv >= config.MIN_AVG_DOLLAR_VOLUME_20D
    checks.append((ok, f"20日均成交額 ${dv / 1e6:.1f}M {'≥' if ok else '<'} ${config.MIN_AVG_DOLLAR_VOLUME_20D / 1e6:.0f}M"))

    # -- RS(20) --
    r20_pct = sr.rs20_ranks.get(yf_sym)
    if r20_pct is not None:
        ok = r20_pct >= config.RS20_PERCENTILE_CUTOFF
        checks.append((ok, f"RS(20) 百分位 {r20_pct:.1f} {'≥' if ok else '<'} {config.RS20_PERCENTILE_CUTOFF}"))
    else:
        checks.append((None, "RS(20) 無法計算"))

    # -- RS(60) --
    r60_pct = sr.rs60_ranks.get(yf_sym)
    if r60_pct is not None:
        ok = r60_pct >= config.RS60_PERCENTILE_CUTOFF
        checks.append((ok, f"RS(60) 百分位 {r60_pct:.1f} {'≥' if ok else '<'} {config.RS60_PERCENTILE_CUTOFF}"))
    else:
        checks.append((None, "RS(60) 無法計算"))

    # -- momentum acceleration --
    if r20_pct is not None and r60_pct is not None:
        ok = r20_pct > r60_pct
        checks.append((ok, f"動量加速 RS20({r20_pct:.1f}) {'>' if ok else '≤'} RS60({r60_pct:.1f})"))

    # -- T-values --
    for label, window in config.T_WINDOWS.items():
        tv = calc_t_value(c, h, lo, window)
        if tv is not None:
            info = f"{label} = {tv:.4f}"
            if label == "T60" and not sr.t_values.empty:
                t60_cut = sr.t_values["T60"].quantile(config.T60_PERCENTILE_CUTOFF / 100)
                ok = tv >= t60_cut
                info += f" ({'≥' if ok else '<'} 門檻 {t60_cut:.4f})"
                checks.append((ok, info))
            elif label == "T120" and not sr.t_values.empty:
                med = sr.t_values["T120"].median()
                ok = tv >= med
                info += f" ({'≥' if ok else '<'} 中位數 {med:.4f})"
                checks.append((ok, info))
            else:
                checks.append((None, info))
        else:
            checks.append((None, f"{label} 無法計算"))

    # -- volume ratio --
    vr = calc_volume_ratio(v)
    if vr is not None:
        ok = vr >= 1.0
        checks.append((ok, f"量增 20d/50d = {vr:.2f} {'≥' if ok else '<'} 1.0"))
    else:
        checks.append((None, "量增 無法計算"))

    # -- 52-week high --
    p52 = pct_from_52w_high(c)
    if p52 is not None:
        ok = p52 >= config.NEAR_52W_HIGH_PCT
        checks.append((ok, f"距52週高點 {p52 * 100:.1f}% {'≥' if ok else '<'} {config.NEAR_52W_HIGH_PCT * 100:.0f}%"))
    else:
        checks.append((None, "52週高點 無法計算"))

    # -- quoteType verification --
    verified = verify_common_stocks([yf_sym])
    is_stock = yf_sym in verified
    checks.append((is_stock, f"證券類型: {'普通股 ✓' if is_stock else '非普通股（基金/信託/衍生品）'}"))

    # -- final verdict --
    passed_all = all(c[0] is True for c in checks if c[0] is not None)
    in_top = yf_sym in sr.top.index
    checks.append((in_top, f"最終結果: {'✅ 入選 Top {}'.format(config.TOP_N) if in_top else '❌ 未入選'}"))

    return checks


# ─────────────────────────────────────────────────────────
#  Telegram Formatting Helpers
# ─────────────────────────────────────────────────────────

def format_screening_msg(result: pd.DataFrame, date_str: str) -> str:
    """Format top-N table for Telegram (MarkdownV2)."""
    if result.empty:
        return f"📊 *US RS\\+T Screener — {date_str}*\n\n⚠️ 今日無符合條件的標的"

    lines = [f"📊 *US RS\\+T Screener — {date_str}*", "```"]
    lines.append(f"{'#':<2} {'Ticker':<14} {'RS20':>5} {'RS60':>5} {'T60':>7} {'Score':>6}")
    lines.append("─" * 44)

    for i, (_, row) in enumerate(result.iterrows(), 1):
        tv = row.get("tv_ticker", "?")
        rs20 = f"{row['rs20_pct']:.0f}" if pd.notna(row.get("rs20_pct")) else "-"
        rs60 = f"{row['rs60_pct']:.0f}" if pd.notna(row.get("rs60_pct")) else "-"
        t60 = f"{row['T60']:.3f}" if pd.notna(row.get("T60")) else "-"
        sc = f"{row['score']:.3f}" if pd.notna(row.get("score")) else "-"
        lines.append(f"{i:<2} {tv:<14} {rs20:>5} {rs60:>5} {t60:>7} {sc:>6}")

    lines.append("```")

    # Acceleration flag
    if "T20" in result.columns and "T60" in result.columns:
        accel = result[result["T20"].fillna(0) > result["T60"].fillna(0)]
        if not accel.empty:
            names = ", ".join(result.at[s, "tv_ticker"] for s in accel.index)
            lines.append(f"\n🔥 趨勢加速中: {names}")

    return "\n".join(lines)


def format_forward_msg(perf: pd.DataFrame, sr: ScreenResult) -> str:
    """Format 3-month forward performance table."""
    if perf is None or perf.empty:
        return "⚠️ 無可用的前瞻數據"

    lines = ["", "📈 *3個月前瞻表現*", "```"]
    lines.append(f"{'#':<2} {'Ticker':<14} {'最高%':>7} {'MDD%':>7} {'天數':>4}")
    lines.append("─" * 38)

    for i, (sym, row) in enumerate(perf.iterrows(), 1):
        tv = sr.top.at[sym, "tv_ticker"] if sym in sr.top.index else sym
        mr = f"+{row['max_ret_pct']:.1f}" if row.get("max_ret_pct", 0) >= 0 else f"{row['max_ret_pct']:.1f}"
        md = f"-{row['mdd_pct']:.1f}"
        days = str(int(row.get("fwd_days", 0)))
        lines.append(f"{i:<2} {tv:<14} {mr:>7} {md:>7} {days:>4}")

    lines.append("```")

    fwd_n = int(perf["fwd_days"].max()) if "fwd_days" in perf.columns else 0
    if fwd_n < config.FORWARD_TRADING_DAYS:
        lines.append(f"\n⚠️ 僅有 {fwd_n}/{config.FORWARD_TRADING_DAYS} 個交易日的前瞻數據")

    return "\n".join(lines)


def format_diagnose_msg(
    symbol: str,
    checks: list[tuple[bool | None, str]],
    date_str: str,
) -> str:
    """Format symbol diagnosis as a readable checklist."""
    lines = [f"🔍 *{symbol} 檢查清單 — {date_str}*", ""]
    for passed, desc in checks:
        if passed is True:
            icon = "✅"
        elif passed is False:
            icon = "❌"
        else:
            icon = "➖"
        lines.append(f"{icon} {desc}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    sr = run_screening()
    if not sr.top.empty:
        print(sr.top.to_string())
        print(f"\nTXT → {sr.txt_path}")
    else:
        print("No results.")
