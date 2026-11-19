#!/usr/bin/env python3
"""gta6-trophy-rarity-forecast-01 entry.

Forecasts, for each of GTA VI's 55 PS5 trophies, five quantiles of its earn
rate on Exophase at the case's target, scored by pinball loss on the logit
scale. Python 3.13, standard library only.

Model (two paths, chosen by what the case's history shows):

1. Trend path -- the history holds >= 3 distinct snapshots (a snapshot is a
   read whose players-tracked count or any rate changed from the previous
   read). Per trophy, an OLS line of logit(rate) on ln(1 + hours since
   2026-11-19T00:00Z) over the last 24 hours of distinct snapshots only,
   anchored at the last read, with slopes shrunk toward the cross-trophy
   median (k=3) and the logit shift clipped to [-3.5, +4.5]. The launch-night
   dilution regime gives way to a progression regime, so all-history fits
   are stale.

2. Dilution-prior path -- the history is flat (< 3 distinct snapshots, the
   practice cases). The page has been unchanged for the whole flat stretch;
   the competition's own record says the first refresh came after ~4.5 flat
   hours, multiplying tracked players 638 -> 3,868 and taking every
   non-intro rate to ~22% of its old value. Expect
   k = (flat stretch + horizon) / 4.5h refreshes (capped at 2) and dilute
   each non-intro rate by 0.2208**k. A trophy at 100% stays at 100%.

Spread (logit): s = sqrt(0.4^2 + (0.5*sqrt(days))^2 + resid^2), capped at
1.6, where resid is the fit residual (trend path) or 0.5 (dilution path).
Quantiles: q_tau = 100 / (1 + exp(-(logit(median) + z_tau * s))) with
z = (-1.645, -0.674, 0, 0.674, 1.645). A trophy at 100% is forecast at
exactly 100% with no spread.

Reads every *.in in $MAKEFUN_INPUT_DIR (else /input/), writes one
/output/<id>.out per case to $MAKEFUN_OUTPUT_DIR (else /output/).
Deterministic: the same case always writes the same bytes.
"""

import json
import math
import os
from datetime import datetime, timezone

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# --- model constants (fixed, never tuned on the practice answers) ---
SHRINK_K = 3.0          # slope shrinkage toward the cross-trophy median
SHIFT_LO = -3.5         # clip on the extrapolated logit shift
SHIFT_HI = 4.5
WINDOW_HOURS = 24.0     # trend fit uses only the last 24h of snapshots
BASE_SPREAD = 0.4       # base term of the logit spread
RW_SPREAD = 0.5         # random-walk term per sqrt(day)
SPREAD_CAP = 1.6
DILUTION_PER_REFRESH = 0.2208   # non-intro rate after one Exophase refresh
REFRESH_HOURS = 4.5     # flat hours per refresh, on the documented record
MAX_REFRESHES = 2.0
DILUTION_RESID = 0.5    # residual term for the dilution-prior path


def _parse_time(s):
    """Parse an ISO 8601 timestamp (with 'Z' or an offset) to a datetime."""
    if s is None:
        return None
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _hours_since_release(dt):
    return (dt - RELEASE).total_seconds() / 3600.0


def _log_time(t):
    """ln(1 + hours since release), floored so pre-release reads stay valid
    (real reads start ~10 minutes before release, t >= -0.17, but never trust
    the input to stay above -1)."""
    return math.log1p(max(t, -0.9))


def _clamp_pct(x):
    return min(99.99, max(0.01, x))


def _logit(pct):
    p = _clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def _inv_logit(lg):
    return 100.0 / (1.0 + math.exp(-lg))


def _round4(x):
    return round(x, 4)


def _ols(xs, ys):
    """Least squares y ~ a + b*x. Returns (slope, intercept, resid_std)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0.0:
        return 0.0, my, 0.0
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    resid = math.sqrt(
        sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / max(n - 2, 1)
    )
    return b, a, resid


def _spread(days, resid):
    s = math.sqrt(
        BASE_SPREAD ** 2 + (RW_SPREAD * math.sqrt(days)) ** 2 + resid ** 2
    )
    return min(s, SPREAD_CAP)


def _quantiles(median_pct, spread):
    """Five quantiles around a median percent, or all-100 for a 100% trophy."""
    if median_pct >= 100.0:
        return [100.0] * len(Z)
    lg = _logit(median_pct)
    out = []
    prev = -1.0
    for z in Z:
        q = _round4(_inv_logit(lg + z * spread))
        q = min(100.0, max(0.0, q))
        # Rounding must never break the non-decreasing order.
        q = max(q, prev)
        out.append(q)
        prev = q
    return out


def _read_times(case):
    """Hours since release for each history read.

    Real cases carry 'at' (falling back to 'slot'); if a read lacks both,
    fill it by interpolation between its timed neighbours, extrapolating
    with a 4.5h spacing at the ends.
    """
    times = []
    for r in case["history"]:
        dt = _parse_time(r.get("at")) or _parse_time(r.get("slot"))
        times.append(None if dt is None else _hours_since_release(dt))
    n = len(times)
    known = [i for i, t in enumerate(times) if t is not None]
    if not known:
        t_last = _hours_since_release(_parse_time(case["at"]))
        return [t_last - (n - 1 - j) * REFRESH_HOURS for j in range(n)]
    # Gaps between known times, for interpolation/extrapolation.
    gaps = [
        times[known[i + 1]] - times[known[i]]
        for i in range(len(known) - 1)
    ]
    gap = sorted(gaps)[len(gaps) // 2] if gaps else REFRESH_HOURS
    if gap <= 0:
        gap = REFRESH_HOURS
    out = list(times)
    for j in range(n):
        if out[j] is not None:
            continue
        after = next((i for i in known if i > j), None)
        before = next((i for i in reversed(known) if i < j), None)
        if before is not None and after is not None:
            frac = (j - before) / (after - before)
            out[j] = times[before] + frac * (times[after] - times[before])
        elif after is not None:
            out[j] = times[after] - (after - j) * gap
        else:
            out[j] = times[before] + (j - before) * gap
    return out


def _distinct_snapshots(case, times):
    """Reads whose players count or any rate changed from the previous read.

    Returns a list of (time_hours, read) in history order.
    """
    snaps = []
    prev_key = None
    for r, t in zip(case["history"], times):
        key = (r.get("players_tracked"), tuple(r["rates"]))
        if key != prev_key:
            snaps.append((t, r))
            prev_key = key
    return snaps


def _last_non_null_rate(case, idx):
    """Most recent non-null rate for trophy idx, or None if never observed."""
    for r in reversed(case["history"]):
        v = r["rates"][idx]
        if v is not None:
            return v
    return None


def forecast_case(case, case_id=None):
    """Forecast one case. Returns {"trophies": [{"id": n, "q": [...]}]}."""
    trophies = case["trophies"]
    n_tr = len(trophies)
    horizon_days = case["horizon_minutes"] / 1440.0
    horizon_hours = case["horizon_minutes"] / 60.0
    t_last = _hours_since_release(_parse_time(case["at"]))
    t_tgt = _hours_since_release(_parse_time(case["target"]))
    x_last = _log_time(t_last)
    x_tgt = _log_time(t_tgt)

    times = _read_times(case)
    snaps = _distinct_snapshots(case, times)
    last_rates = [_last_non_null_rate(case, i) for i in range(n_tr)]

    rows = []
    if len(snaps) >= 3:
        # ---- Trend path: OLS logit(rate) on log1p(hours) over the last 24h.
        window = [(t, r) for (t, r) in snaps if t >= t_last - WINDOW_HOURS]
        if len(window) < 3:
            window = snaps
        xs = [_log_time(t) for (t, _r) in window]

        # Per-trophy fit (null rates skipped for that trophy only).
        fits = []
        slopes = []
        for i in range(n_tr):
            pts = [
                (x, _logit(r["rates"][i]))
                for x, (_t, r) in zip(xs, window)
                if r["rates"][i] is not None
            ]
            if len(pts) >= 2:
                b, _a, resid = _ols([p[0] for p in pts], [p[1] for p in pts])
                fits.append((b, len(pts), min(resid, 1.0)))
                slopes.append(b)
            else:
                fits.append((None, len(pts), DILUTION_RESID))
        med_slope = sorted(slopes)[len(slopes) // 2] if slopes else 0.0

        for i, t in enumerate(trophies):
            anchor = last_rates[i]
            if anchor is None:
                anchor = 1.0
            if anchor >= 99.99:
                rows.append({"id": t["id"], "q": [100.0] * len(Z)})
                continue
            b, npts, resid = fits[i]
            if b is None:
                b = med_slope
            else:
                b = (npts * b + SHRINK_K * med_slope) / (npts + SHRINK_K)
            shift = b * (x_tgt - x_last)
            shift = min(SHIFT_HI, max(SHIFT_LO, shift))
            median = _clamp_pct(_inv_logit(_logit(anchor) + shift))
            rows.append(
                {"id": t["id"], "q": _quantiles(median, _spread(horizon_days, resid))}
            )
    else:
        # ---- Dilution-prior path: flat history, refreshes yet to come.
        # Flat stretch = time the page has been unchanged, from the first
        # read we hold to the checkpoint.
        flat_hours = t_last - times[0] if times else 0.0
        k = (flat_hours + horizon_hours) / REFRESH_HOURS
        k = min(max(k, 0.0), MAX_REFRESHES)
        factor = DILUTION_PER_REFRESH ** k
        spread = _spread(horizon_days, DILUTION_RESID)
        for i, t in enumerate(trophies):
            anchor = last_rates[i]
            if anchor is None:
                anchor = 1.0
            if anchor >= 99.99:
                rows.append({"id": t["id"], "q": [100.0] * len(Z)})
                continue
            median = _clamp_pct(anchor * factor)
            rows.append({"id": t["id"], "q": _quantiles(median, spread)})

    return {"trophies": rows}


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(output_dir, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    names = sorted(f for f in os.listdir(input_dir) if f.endswith(".in"))
    for name in names:
        case_id = name[: -len(".in")]
        with open(os.path.join(input_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        if case.get("levels") != LEVELS:
            raise ValueError(
                f"{name} asks for levels {case.get('levels')}, not {LEVELS}"
            )
        if not case.get("history"):
            raise ValueError(f"{name} has no history")
        forecast = forecast_case(case, case_id)
        out_path = os.path.join(output_dir, f"{case_id}.out")
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(forecast) + "\n")


if __name__ == "__main__":
    main()
