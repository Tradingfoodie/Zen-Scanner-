"""ZEN ORB TEST: does a 30-minute opening range breakout follow through?

For each stock and day (last ~60 days of 5-minute bars, regular hours):
  - Opening range (OR) = 9:30-10:00 ET (8:30-9:00 Central): its high, low and height R.
  - Entry = close of the FIRST 5-minute bar that closes above the OR high (long) or
    below the OR low (short), any time after 10:00 ET. One trade per stock per day.
  - Stop = the middle of the OR. Target = 1 OR height (R) beyond the entry.
  - Result in units of risk (entry-to-stop): target first = +reward/risk, stop first = -1,
    neither by the close = wherever it closed. A bar that touches both counts as the stop.
Also measures the best move after entry (sell-into-momentum) and how often it fell back
inside the range. Research only. Yahoo data; 5-minute bars keep about 60 days.
"""
import datetime as dt

import numpy as np
import pandas as pd

OR_MIN = 30


def day_trades(df, or_min=30):
    out = []
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    if df.empty:
        return out
    idx = df.index.tz_convert("America/New_York")
    df = df.set_index(idx)
    df = df.between_time("09:30", "15:55")
    for day, d in df.groupby(df.index.date):
        if len(d) < 60:
            continue
        end = (dt.datetime(2000, 1, 1, 9, 30) + dt.timedelta(minutes=or_min))
        orw = d.between_time("09:30", (end - dt.timedelta(minutes=5)).strftime("%H:%M"))
        rest = d.between_time(end.strftime("%H:%M"), "15:55")
        if len(orw) < or_min // 5 or rest.empty:
            continue
        hi, lo = float(orw["High"].max()), float(orw["Low"].min())
        R = hi - lo
        if R <= 0:
            continue
        mid = (hi + lo) / 2
        up = rest["Close"] > hi
        dn = rest["Close"] < lo
        first_up = up.idxmax() if up.any() else None
        first_dn = dn.idxmax() if dn.any() else None
        if first_up is None and first_dn is None:
            continue
        if first_dn is None or (first_up is not None and first_up < first_dn):
            side, t0 = 1, first_up
        else:
            side, t0 = -1, first_dn
        entry = float(rest.loc[t0, "Close"])
        after = rest[rest.index > t0]
        risk = abs(entry - mid)
        if risk <= 0:
            continue
        target = entry + side * R
        res, how = None, "close"
        for _, b in after.iterrows():
            hit_stop = b["Low"] <= mid if side == 1 else b["High"] >= mid
            hit_tgt = b["High"] >= target if side == 1 else b["Low"] <= target
            if hit_stop:
                res, how = -1.0, "stop"
                break
            if hit_tgt:
                res, how = R / risk, "target"
                break
        last = float(d["Close"].iloc[-1])
        if res is None:
            res = side * (last - entry) / risk
        if after.empty:
            best = worst = 0.0
        elif side == 1:
            best, worst = float(after["High"].max()) - entry, entry - float(after["Low"].min())
        else:
            best, worst = entry - float(after["Low"].min()), float(after["High"].max()) - entry
        back_inside = bool(((after["Close"] < hi) & (after["Close"] > lo)).any()) if not after.empty else False
        out.append(dict(day=day, side=side, t=t0.time(), entry=entry, R=R, risk=risk, res=res, how=how,
                        best_pct=best / entry, worst_pct=worst / entry, best_R=best / R,
                        close_pct=side * (last - entry) / entry, back_inside=back_inside,
                        or_vol=float(orw["Volume"].sum()), or_pct=R / entry))
    return out


def summary(g, title):
    if len(g) < 10:
        return f"{title}: too few ({len(g)})\n"
    tgt = (g["how"] == "target").mean()
    stp = (g["how"] == "stop").mean()
    lines = [title,
             f"   trades:                    {len(g)}",
             f"   hit target (+1 range):     {tgt:.0%}   stopped at mid: {stp:.0%}",
             f"   avg result per trade:      {g['res'].mean():+.2f}R  (0 = break-even)",
             f"   best move after entry:     median {g['best_pct'].median():+.2%} | ran 1%+: {(g['best_pct'] >= .01).mean():.0%}",
             f"   fell back inside range:    {g['back_inside'].mean():.0%}",
             f"   held to close:             {g['close_pct'].mean():+.2%} avg",
             ""]
    return "\n".join(lines) + "\n"


def build(data, tick, market, or_min):
    rows, spy_dir = [], {}
    for t in tick:
        try:
            df = data[t]
        except Exception:
            continue
        trades = day_trades(df, or_min)
        for tr in trades:
            tr["ticker"] = t
        if t == "SPY":
            spy_dir = {tr["day"]: tr["side"] for tr in trades}
        if t not in market:
            rows.extend(trades)
    if not rows:
        return None
    ev = pd.DataFrame(rows).sort_values(["day", "ticker"])
    ev["rvol"] = ev.groupby("ticker")["or_vol"].transform(lambda s: s / s.rolling(10, min_periods=5).mean().shift(1))
    ev["spy"] = ev["day"].map(spy_dir)
    return ev


def run_volume(yf, tickers, market=("SPY", "QQQ")):
    """Follow-up: high-volume opens only, both halves, 15 vs 30-minute ranges."""
    tick = sorted(set(list(tickers) + list(market)))
    data = yf.download(tick, period="60d", interval="5m", group_by="ticker", auto_adjust=False,
                       prepost=False, progress=False, threads=True)
    out = ["ZEN ORB FOLLOW-UP: high-volume opens (opening-range volume 1.5x+ the stock's 10-day average)\n"]
    for or_min in (15, 30):
        ev = build(data, tick, market, or_min)
        if ev is None:
            return "No data came back. Yahoo may be throttling; try again later."
        days = sorted(ev["day"].unique())
        half = days[len(days) // 2]
        hv = ev[ev["rvol"] >= 1.5]
        out.append(f"===== {or_min}-MINUTE RANGE ({days[0]} to {days[-1]}) =====\n")
        out.append(summary(ev, f"All breakouts ({or_min} min), for comparison"))
        out.append(summary(hv, "HIGH VOLUME, all"))
        out.append(summary(hv[hv["day"] < half], "HIGH VOLUME, first half"))
        out.append(summary(hv[hv["day"] >= half], "HIGH VOLUME, second half"))
        out.append(summary(hv[hv["spy"] == hv["side"]], "HIGH VOLUME + same direction as SPY"))
        out.append(summary(ev[ev["rvol"] >= 2.0], "VERY HIGH VOLUME (2x+)"))
    out.append("Edge = above +0.10R in BOTH halves. Anything less is not worth an alert.")
    return "\n".join(out)


def run(yf, tickers, market=("SPY", "QQQ")):
    tick = sorted(set(list(tickers) + list(market)))
    data = yf.download(tick, period="60d", interval="5m", group_by="ticker", auto_adjust=False,
                       prepost=False, progress=False, threads=True)
    rows, spy_dir = [], {}
    for t in tick:
        try:
            df = data[t]
        except Exception:
            continue
        trades = day_trades(df)
        for tr in trades:
            tr["ticker"] = t
        if t == "SPY":
            spy_dir = {tr["day"]: tr["side"] for tr in trades}
        if t not in market:
            rows.extend(trades)
    if not rows:
        return "No data came back. Yahoo may be throttling; try again later."
    ev = pd.DataFrame(rows).sort_values(["day", "ticker"])
    # Opening-range volume vs that stock's own average (RVOL), and SPY's break direction.
    ev["rvol"] = ev.groupby("ticker")["or_vol"].transform(lambda s: s / s.rolling(10, min_periods=5).mean().shift(1))
    ev["spy"] = ev["day"].map(spy_dir)
    days = sorted(ev["day"].unique())
    half = days[len(days) // 2]
    out = [f"ZEN ORB TEST (30-min opening range, {len(days)} days, {ev['ticker'].nunique()} stocks)",
           f"{days[0]} to {days[-1]} | entry = first 5-min close outside the range",
           "stop = range middle | target = +1 range height\n"]
    out.append(summary(ev, "ALL BREAKOUTS"))
    out.append(summary(ev[ev["side"] == 1], "LONG (broke up)"))
    out.append(summary(ev[ev["side"] == -1], "SHORT (broke down)"))
    out.append(summary(ev[ev["rvol"] >= 1.5], "HIGH VOLUME open (1.5x+ normal)"))
    out.append(summary(ev[ev["spy"] == ev["side"]], "SAME DIRECTION AS SPY's break"))
    out.append(summary(ev[(ev["spy"].notna()) & (ev["spy"] != ev["side"])], "AGAINST SPY's break"))
    t = pd.to_datetime(ev["t"].astype(str), format="%H:%M:%S").dt.time
    out.append(summary(ev[t < dt.time(11, 0)], "BROKE OUT EARLY (before 10:00 Central)"))
    out.append(summary(ev[t >= dt.time(11, 0)], "BROKE OUT LATER (10:00 Central or after)"))
    out.append(summary(ev[ev["day"] < half], "FIRST HALF of the period"))
    out.append(summary(ev[ev["day"] >= half], "SECOND HALF (check it holds)"))
    per = ev.groupby("ticker").agg(n=("res", "size"), avg=("res", "mean"))
    per = per[per["n"] >= 15].sort_values("avg", ascending=False)
    if len(per):
        out.append("BEST ORB STOCKS (15+ trades)")
        out += [f"   {k:<5} {r['avg']:+.2f}R over {int(r['n'])}" for k, r in per.head(8).iterrows()]
        out.append("\nWORST")
        out += [f"   {k:<5} {r['avg']:+.2f}R over {int(r['n'])}" for k, r in per.tail(5).iloc[::-1].iterrows()]
    out.append("\nHow to read it: avg result above +0.10R in BOTH halves would be")
    out.append("a real edge worth building a morning alert for. Near 0 = coin flip.")
    out.append("Options add leverage but also cost (spread, time decay), so a stock")
    out.append("edge has to be clearly positive to survive as an options trade.")
    return "\n".join(out)
