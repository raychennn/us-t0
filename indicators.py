"""
Core Indicators
═══════════════
1. RS  — dual-period relative strength  (20-day & 60-day)
2. T   — trend quality score  (from the 「勢」 framework)
3. Aux — volume ratio, volatility, 52-week-high proximity
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import ATR_PERIOD, T_MA_PERIOD, T_WEIGHTED_BLEND

# ═══════════════════════════════════════════════════════════
#  RS — Relative Strength
# ═══════════════════════════════════════════════════════════


def calc_rs_return(close: np.ndarray, period: int) -> float | None:
    """
    Simple price return over *period* trading days.

        RS = close[-1] / close[-1 - period] - 1

    Returns None if there isn't enough data.
    """
    needed = period + 1
    if len(close) < needed:
        return None
    p_now = close[-1]
    p_past = close[-1 - period]
    if p_past <= 0:
        return None
    return float(p_now / p_past - 1.0)


def rank_percentile(series: pd.Series) -> pd.Series:
    """Convert raw values → 0-99 percentile ranks."""
    return series.rank(pct=True).mul(99).round(1)


# ═══════════════════════════════════════════════════════════
#  T — Trend Quality
# ═══════════════════════════════════════════════════════════

# ---------- helpers ----------

def _sma(values: np.ndarray, period: int) -> np.ndarray:
    """Simple moving average (NaN-filled head)."""
    return pd.Series(values).rolling(period).mean().to_numpy()


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
         period: int) -> np.ndarray:
    """Average True Range (NaN-filled head)."""
    n = len(close)
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                     abs(high[i] - close[i - 1]),
                     abs(low[i] - close[i - 1]))
    return pd.Series(tr).rolling(period).mean().to_numpy()


# ---------- standardisation ----------

def _standardise(close: np.ndarray, ma_period: int = T_MA_PERIOD
                 ) -> np.ndarray:
    """
    Map daily price movements to a displacement sequence S.

    Rules (from the article):
      cross up   → σ = +1
      cross down → σ = -1
      above MA, up   → +1;  above MA, down → 0
      below MA, down → -1;  below MA, up   → 0

    Returns cumulative displacement S (same length as close).
    """
    n = len(close)
    if n < ma_period + 2:
        return np.zeros(n)

    ma = _sma(close, ma_period)
    sigma = np.zeros(n)

    for k in range(ma_period, n):
        prev_above = close[k - 1] >= ma[k - 1]
        curr_above = close[k] >= ma[k]

        if not prev_above and curr_above:
            sigma[k] = 1.0
        elif prev_above and not curr_above:
            sigma[k] = -1.0
        elif prev_above and curr_above:
            sigma[k] = 1.0 if close[k] > close[k - 1] else 0.0
        else:
            sigma[k] = -1.0 if close[k] < close[k - 1] else 0.0

    return np.cumsum(sigma)


def _standardise_weighted(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    ma_period: int = T_MA_PERIOD,
    atr_period: int = ATR_PERIOD,
) -> np.ndarray:
    """
    ATR-weighted displacement:  σ_w(k) = direction × |Δclose| / ATR.
    Preserves magnitude information lost by the ±1 binary encoding.
    """
    n = len(close)
    start = max(ma_period, atr_period)
    if n < start + 2:
        return np.zeros(n)

    ma = _sma(close, ma_period)
    atr_vals = _atr(high, low, close, atr_period)
    sigma_w = np.zeros(n)

    for k in range(start, n):
        if np.isnan(atr_vals[k]) or atr_vals[k] <= 0:
            continue

        prev_above = close[k - 1] >= ma[k - 1]
        curr_above = close[k] >= ma[k]

        if not prev_above and curr_above:
            direction = 1.0
        elif prev_above and not curr_above:
            direction = -1.0
        elif prev_above and curr_above:
            direction = 1.0 if close[k] > close[k - 1] else 0.0
        else:
            direction = -1.0 if close[k] < close[k - 1] else 0.0

        sigma_w[k] = direction * abs(close[k] - close[k - 1]) / atr_vals[k]

    return np.cumsum(sigma_w)


# ---------- T1: consecutive-segment method ----------

def _turning_points(S: np.ndarray) -> list[int]:
    """Indices of local extrema in S (always includes first & last)."""
    n = len(S)
    if n < 3:
        return list(range(n))

    pts = [0]
    for i in range(1, n - 1):
        if (S[i] >= S[i - 1] and S[i] >= S[i + 1]
                and not (S[i] == S[i - 1] == S[i + 1])):
            pts.append(i)
        elif (S[i] <= S[i - 1] and S[i] <= S[i + 1]
              and not (S[i] == S[i - 1] == S[i + 1])):
            pts.append(i)
    pts.append(n - 1)

    # de-duplicate consecutive equal-valued turning points
    deduped = [pts[0]]
    for p in pts[1:]:
        if S[p] != S[deduped[-1]]:
            deduped.append(p)
    return deduped


def _calc_t1(S: np.ndarray) -> float:
    """T1 = Σ |S'(i) − S'(i−1)|² over turning points."""
    tp = _turning_points(S)
    return sum(
        (S[tp[i]] - S[tp[i - 1]]) ** 2
        for i in range(1, len(tp))
    )


# ---------- T2: recursive swing decomposition ----------

def _decompose_swings(S: np.ndarray, lo: int, hi: int,
                      deltas: list[float]) -> None:
    """Recursively decompose the largest swings."""
    if hi - lo < 1:
        return
    sub = S[lo: hi + 1]
    max_idx = lo + int(np.argmax(sub))
    min_idx = lo + int(np.argmin(sub))
    delta = abs(S[max_idx] - S[min_idx])
    if delta < 1e-9:
        return
    deltas.append(delta)
    a, b = sorted([max_idx, min_idx])
    if a > lo:
        _decompose_swings(S, lo, a, deltas)
    if b < hi:
        _decompose_swings(S, b, hi, deltas)


def _calc_t2(S: np.ndarray) -> float:
    """T2 = Σ Δ(i)² from recursive swing decomposition."""
    deltas: list[float] = []
    _decompose_swings(S, 0, len(S) - 1, deltas)
    return sum(d ** 2 for d in deltas)


# ---------- composite T-value ----------

def calc_t_value(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    window: int,
    ma_period: int = T_MA_PERIOD,
) -> float | None:
    """
    Composite trend-quality score for the most recent *window* bars.

        T = blend × T_std + (1 − blend) × T_weighted
        where T_* = max(T1, T2) / N^1.5
    """
    lookback = window + ma_period + ATR_PERIOD + 10
    if len(close) < lookback:
        return None

    c = close[-lookback:]
    h = high[-lookback:]
    lo_ = low[-lookback:]
    N = window

    # Standard T
    S = _standardise(c, ma_period)[-window:]
    S = S - S[0]
    t_std = max(_calc_t1(S), _calc_t2(S)) / (N ** 1.5)

    # ATR-weighted T
    Sw = _standardise_weighted(c, h, lo_, ma_period)[-window:]
    Sw = Sw - Sw[0]
    t_wtd = max(_calc_t1(Sw), _calc_t2(Sw)) / (N ** 1.5)

    return T_WEIGHTED_BLEND * t_std + (1 - T_WEIGHTED_BLEND) * t_wtd


# ═══════════════════════════════════════════════════════════
#  Auxiliary Indicators
# ═══════════════════════════════════════════════════════════

def calc_hist_volatility(close: np.ndarray, period: int = 20
                         ) -> float | None:
    """Annualised historical volatility (log-return std × √252)."""
    if len(close) < period + 1:
        return None
    log_ret = np.diff(np.log(close[-(period + 1):]))
    return float(np.std(log_ret, ddof=1) * np.sqrt(252))


def calc_volume_ratio(volume: np.ndarray) -> float | None:
    """20-day average volume ÷ 50-day average volume."""
    if len(volume) < 50:
        return None
    avg50 = np.mean(volume[-50:])
    if avg50 <= 0:
        return None
    return float(np.mean(volume[-20:]) / avg50)


def pct_from_52w_high(close: np.ndarray) -> float | None:
    """Current price as a fraction of the 252-day high."""
    if len(close) < 252:
        return None
    high_52w = np.max(close[-252:])
    if high_52w <= 0:
        return None
    return float(close[-1] / high_52w)
