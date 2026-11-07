"""Zen Futures: the same Potter box rules as the stock scanner, run on 15-minute and hourly
futures bars for Asia-session day trades (MES / MNQ / MGC, full-size ES / NQ / GC alongside).

Trade card for a box breakout that is retesting its broken edge:
    entry  = the broken edge (a limit order on the punchback)
    stop   = the box's 50% mark
    target = the near edge of the next box beyond the break (box to box), else the box
             height projected past the edge
Risk and reward are shown in points, ticks and dollars per contract.

Review: each logged setup is graded on 15-minute bars from the scan to the end of that
Asia session (2:00 AM Central, the London open). Filled = price trades back to the entry.
After a fill, the stop or the target, whichever is touched first, decides it (a bar that
touches both counts as the stop). Still open at 2:00 AM = closed at that bar's price.

Data: Yahoo continuous contracts (ES=F, NQ=F, GC=F), ~10-15 min delayed. Contract rolls
make price jump on the continuous chart, so setups next to a jump are skipped.
Research tool, not trade calls."""
import datetime as dt
import math
import os
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import zen_scanner as z

CT = ZoneInfo("America/Chicago")
LOG_FILE = "zen_futures_log.csv"
SESSION_END_HOUR = 2          # Asia day trades are closed at 2:00 AM Central (London open)
GAP_ATR = 4.0                 # an open this many ATRs away from the prior close = roll/halt jump
DEDUP_HOURS = 24              # the same box is not logged twice within this many hours

# micro, full, Yahoo symbol, $ per point (micro), $ per point (full), tick size
CONTRACTS = [
    ("MES", "ES", "ES=F", 5.0, 50.0, 0.25),
    ("MNQ", "NQ", "NQ=F", 2.0, 20.0, 0.25),
    ("MGC", "GC", "GC=F", 10.0, 100.0, 0.10),
]
FRAMES = [("15m", "15m", "60d"), ("1h", "60m", "180d")]   # label, Yahoo interval, history

LAST_RUN = {}       # structured results of the latest futures scan (used by the website)
LAST_REVIEW = None  # graded log of the latest futures review


# ---- data -----------------------------------------------------------------------
def to_ct(df):
    idx = pd.DatetimeIndex(df.index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx
    df = df.copy()
    df.index = idx.tz_convert(CT)
    return df


def download(interval, period):
    """{yahoo symbol: OHLC DataFrame in Central time} for all contracts."""
    syms = [c[2] for c in CONTRACTS]
    data = z.yf.download(syms, period=period, interval=interval, group_by="ticker",
                         auto_adjust=False, progress=False, threads=True)
    out = {}
    for s in syms:
        try:
            df = data[s][["Open", "High", "Low", "Close"]].dropna()
        except Exception:
            continue
        if len(df) >= 80:
            out[s] = to_ct(df)
    return out


def atr_series(df):
    c, h, l = df["Close"], df["High"], df["Low"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(14).mean()


def has_jump(df, bars):
    """True if any bar in the last `bars` opened GAP_ATR+ ATRs from the prior close:
    a contract roll on the continuous chart (or a halt gap). Boxes across it are not real."""
    a = atr_series(df)
    gap = (df["Open"] - df["Close"].shift()).abs()
    recent = (gap / a.shift()).iloc[-bars:]
    return bool((recent > GAP_ATR).any())


def tick_round(x, tick):
    return round(round(x / tick) * tick, 6)


# ---- scan -----------------------------------------------------------------------
def setup_card(micro, full, mpv, fpv, tick, tf, df, sig):
    d = sig["dir"]
    edge = tick_round(sig["top"] if d == 1 else sig["bot"], tick)
    stop = tick_round(sig["mid"], tick)
    target = tick_round(sig["target"], tick)
    risk = abs(edge - stop)
    reward = (target - edge) * d
    if risk <= 0 or reward <= 0:
        return None
    price = float(df["Close"].iloc[-1])
    return dict(contract=micro, full=full, tf=tf, side="long" if d == 1 else "short", dir=d,
                entry=edge, stop=stop, target=target, top=float(sig["top"]), bot=float(sig["bot"]),
                kind=sig["kind"], state=sig["state"], age=int(sig["age"]), price=price,
                risk_pts=risk, reward_pts=reward, r=reward / risk, risk_ticks=int(round(risk / tick)),
                risk_micro=risk * mpv, risk_full=risk * fpv,
                reward_micro=reward * mpv, reward_full=reward * fpv,
                bar_time=df.index[-1].isoformat())


def scan_contract(spec, frames):
    micro, full, sym, mpv, fpv, tick = spec
    setups, watches = [], []
    for tf, df in frames:
        if df is None or len(df) < 80:
            continue
        if has_jump(df, 60):
            watches.append(dict(contract=micro, tf=tf, status="skip",
                                note="price jump (contract roll?) in the last 60 bars"))
            continue
        sig = z.box_signal(df)
        if sig and sig["state"] in ("retest", "extended"):
            card = setup_card(micro, full, mpv, fpv, tick, tf, df, sig)
            if card:
                setups.append(card)
        box = z.current_box(df)
        if box:
            watches.append(dict(contract=micro, tf=tf, status=box["status"], top=box["top"],
                                bot=box["bot"], mid=box["mid"], price=box["price"],
                                bars=box["days"], to_roof=box["to_roof"], to_floor=box["to_floor"]))
    return setups, watches


def run_futures(now, slot):
    """Scan all contracts, print a summary, log new setups. Returns the printed text."""
    frames = {}
    for tf, interval, period in FRAMES:
        try:
            frames[tf] = download(interval, period)
        except Exception as e:
            print(f"{tf} download failed: {e}")
            frames[tf] = {}
    setups, watches = [], []
    for spec in CONTRACTS:
        s, w = scan_contract(spec, [(tf, frames[tf].get(spec[2])) for tf, _, _ in FRAMES])
        setups += s
        watches += w
    setups.sort(key=lambda c: (c["state"] != "retest", -c["r"]))
    LAST_RUN.clear()
    LAST_RUN.update(time=now.isoformat(), slot=slot, setups=setups, watches=watches,
                    contracts=[c[0] for c in CONTRACTS])
    log_setups(setups, now, slot)
    print_futures(setups, watches, now, slot)
    return setups


def print_futures(setups, watches, now, slot):
    print(f"===== FUTURES · ASIA {slot.upper()} · {now:%a %m/%d %I:%M %p} Central =====")
    if not setups:
        print("No box breakouts retesting right now.")
    for i, c in enumerate(setups, 1):
        side = "LONG " if c["dir"] == 1 else "SHORT"
        print(f"{i}. {c['contract']} {c['tf']:>3} {side} {c['state']:<8} "
              f"entry {c['entry']:,.2f}  stop {c['stop']:,.2f}  target {c['target']:,.2f}  "
              f"R {c['r']:.1f}  risk ${c['risk_micro']:,.0f}/micro (${c['risk_full']:,.0f} full)")
    live = [w for w in watches if w["status"] in ("up", "down", "inside")]
    if live:
        print("\nBox watch:")
        for w in live:
            print(f"  {w['contract']} {w['tf']:>3} {w['status']:<6} box {w['bot']:,.2f}-{w['top']:,.2f}"
                  f"  now {w['price']:,.2f}")
    for w in watches:
        if w["status"] == "skip":
            print(f"  {w['contract']} {w['tf']}: skipped, {w['note']}")


# ---- log ------------------------------------------------------------------------
LOG_COLS = ["scan_time", "slot", "contract", "tf", "dir", "entry", "stop", "target", "top", "bot",
            "kind", "state", "price", "r", "grade", "result_r", "fill_time", "exit_time"]


def read_log():
    if not os.path.exists(LOG_FILE):
        return pd.DataFrame(columns=LOG_COLS)
    log = pd.read_csv(LOG_FILE)
    for c in LOG_COLS:
        if c not in log.columns:
            log[c] = np.nan
    for c in ("slot", "contract", "tf", "kind", "state", "grade", "fill_time", "exit_time"):
        log[c] = log[c].astype(object)      # text columns (blank ones load as numbers)
    log["result_r"] = log["result_r"].astype(float)
    return log[LOG_COLS]


def log_setups(setups, now, slot):
    log = read_log()
    rows = []
    for c in setups:
        recent = log[(log["contract"] == c["contract"]) & (log["tf"] == c["tf"])
                     & (log["dir"] == c["dir"])]
        dup = False
        for _, r in recent.iterrows():
            t = dt.datetime.fromisoformat(str(r["scan_time"]))
            if (now - t).total_seconds() < DEDUP_HOURS * 3600 and \
                    math.isclose(float(r["top"]), c["top"], rel_tol=1e-6) and \
                    math.isclose(float(r["bot"]), c["bot"], rel_tol=1e-6):
                dup = True
        if dup:
            continue
        rows.append(dict(scan_time=now.isoformat(), slot=slot, contract=c["contract"], tf=c["tf"],
                         dir=c["dir"], entry=c["entry"], stop=c["stop"], target=c["target"],
                         top=c["top"], bot=c["bot"], kind=c["kind"], state=c["state"],
                         price=c["price"], r=round(c["r"], 3), grade="open"))
    if rows:
        new = pd.DataFrame(rows, columns=LOG_COLS)
        log = new if log.empty else pd.concat([log, new], ignore_index=True)
        log.to_csv(LOG_FILE, index=False)


# ---- review ---------------------------------------------------------------------
def session_end(t):
    """The 2:00 AM Central after scan time t (end of that Asia session)."""
    end = t.replace(hour=SESSION_END_HOUR, minute=0, second=0, microsecond=0)
    return end if end > t else end + dt.timedelta(days=1)


def grade_setup(r, bars, now):
    """bars: 15-minute OHLC (Central). Returns (grade, result_r, fill_time, exit_time)."""
    t0 = dt.datetime.fromisoformat(str(r["scan_time"]))
    end = session_end(t0)
    d, entry, stop, target = int(r["dir"]), float(r["entry"]), float(r["stop"]), float(r["target"])
    risk = abs(entry - stop)
    w = bars[(bars.index > t0) & (bars.index < end)]
    filled = None
    for ts, b in w.iterrows():
        if filled is None:
            if (b["Low"] <= entry) if d == 1 else (b["High"] >= entry):
                filled = ts             # price traded back to the entry (limit fills)
            else:
                continue
        hit_stop = b["Low"] <= stop if d == 1 else b["High"] >= stop
        hit_tgt = b["High"] >= target if d == 1 else b["Low"] <= target
        if hit_stop:
            return "LOSS", -1.0, filled.isoformat(), ts.isoformat()
        if hit_tgt:
            return "WIN", (target - entry) * d / risk, filled.isoformat(), ts.isoformat()
    if now < end:
        return "open", np.nan, filled.isoformat() if filled is not None else "", ""
    if filled is None:
        return "NO FILL", 0.0, "", ""
    last = w.iloc[-1]
    return "TIME", (float(last["Close"]) - entry) * d / risk, filled.isoformat(), w.index[-1].isoformat()


def run_review(now):
    """Grade every open setup in the futures log and print a scorecard."""
    global LAST_REVIEW
    log = read_log()
    if log.empty:
        print("No futures setups logged yet.")
        LAST_REVIEW = log
        return
    todo = log["grade"].fillna("open") == "open"
    if todo.any():
        bars = download("15m", "60d")
        sym = {c[0]: c[2] for c in CONTRACTS}
        for i in log.index[todo]:
            b = bars.get(sym.get(log.at[i, "contract"]))
            if b is None:
                continue
            g, rr, ft, et = grade_setup(log.loc[i], b, now)
            log.loc[i, ["grade", "result_r", "fill_time", "exit_time"]] = [g, rr, ft, et]
        log.to_csv(LOG_FILE, index=False)
    LAST_REVIEW = log
    print_review(log)


def review_stats(log):
    """Per-contract and overall stats on filled setups. $ uses 1 micro contract."""
    pv = {c[0]: (c[3], c[4]) for c in CONTRACTS}
    out = []
    for name, g in [("ALL", log)] + list(log.groupby("contract")):
        filled = g[g["grade"].isin(["WIN", "LOSS", "TIME"])]
        if not len(g):
            continue
        risk = (filled["entry"] - filled["stop"]).abs()
        rr = filled["result_r"].astype(float)
        micro = (rr * risk * filled["contract"].map(lambda c: pv.get(c, (0, 0))[0])).sum()
        out.append(dict(name=name, logged=len(g), filled=len(filled),
                        wins=int((filled["grade"] == "WIN").sum()),
                        losses=int((filled["grade"] == "LOSS").sum()),
                        timed=int((filled["grade"] == "TIME").sum()),
                        nofill=int((g["grade"] == "NO FILL").sum()),
                        open=int((g["grade"].fillna("open") == "open").sum()),
                        total_r=float(rr.sum()) if len(rr) else 0.0,
                        avg_r=float(rr.mean()) if len(rr) else 0.0,
                        dollars_micro=float(micro)))
    return out


def print_review(log):
    print("===== FUTURES REVIEW (Asia day trades, 1 micro each) =====")
    for s in review_stats(log):
        print(f"{s['name']:<4} filled {s['filled']}/{s['logged']}  "
              f"{s['wins']}W {s['losses']}L {s['timed']} timed-out  "
              f"total {s['total_r']:+.1f}R  avg {s['avg_r']:+.2f}R  ${s['dollars_micro']:+,.0f}"
              f"  ({s['nofill']} no fill, {s['open']} open)")
