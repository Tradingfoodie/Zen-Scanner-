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
import json
import math
import os
import urllib.request
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

CHART_BARS = 64              # bars kept for each chart on the website
SITE_URL = "https://tradingfoodie.github.io/Zen-Scanner-/"
RTH = (dt.time(8, 30), dt.time(15, 0))   # New York session, Central time

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


# ---- chart data -----------------------------------------------------------------
def chart_bars(df, n=CHART_BARS):
    """Last n bars as [[open, high, low, close], ...] plus first/last bar time."""
    w = df.iloc[-n:]
    bars = [[round(float(r.Open), 4), round(float(r.High), 4), round(float(r.Low), 4),
             round(float(r.Close), 4)] for r in w.itertuples()]
    return dict(bars=bars, t0=w.index[0].isoformat(), t1=w.index[-1].isoformat())


def box_span(df, top, bot, end_age):
    """Find the box window (start, end bar offsets from the right) that box_signal used,
    so the chart can shade exactly those bars. end_age = bars since the box ended."""
    o, c = df["Open"], df["Close"]
    bt, bb = np.maximum(o, c), np.minimum(o, c)
    e = end_age
    for L in (40, 30, 20, 15, 10):
        if e + L > len(df):
            continue
        if math.isclose(float(bt.iloc[-(e + L):-e].max()), top, rel_tol=1e-9) and \
                math.isclose(float(bb.iloc[-(e + L):-e].min()), bot, rel_tol=1e-9):
            return e + L, e
    return e + 20, e


# ---- key levels -----------------------------------------------------------------
def session_ids(idx):
    """CME trading day of each bar: the Globex day starts at 5:00 PM Central, so
    shifting by 7 hours puts Sunday 5 PM - Monday 4 PM on Monday."""
    return (idx + pd.Timedelta(hours=7)).date


def key_levels(df):
    """Levels a night trader watches, from 15-minute bars in Central time:
    prior Globex day high/low, the last New York session high/low/close (8:30-3:00),
    and tonight's high/low since 5:00 PM. Plus the nearest level above and below price."""
    if df is None or len(df) < 50:
        return None
    sid = pd.Series(session_ids(df.index), index=df.index)
    days = sorted(set(sid))
    cur = days[-1]
    tm = pd.Series([t.time() for t in df.index], index=df.index)
    rth_mask = (tm >= RTH[0]) & (tm < RTH[1])
    px = float(df["Close"].iloc[-1])
    lv = []
    if len(days) >= 2:
        prev = df[sid == days[-2]]
        lv += [("Prior day high", float(prev["High"].max())), ("Prior day low", float(prev["Low"].min()))]
    rth = df[rth_mask]
    if len(rth):
        last_day = rth.index[-1].date()
        ny = rth[[t.date() == last_day for t in rth.index]]
        lv += [("NY high", float(ny["High"].max())), ("NY low", float(ny["Low"].min())),
               ("NY close", float(ny["Close"].iloc[-1]))]
    tonight = df[(sid == cur) & ~rth_mask & ((tm >= dt.time(17, 0)) | (tm < RTH[0]))]
    if len(tonight):
        lv += [("Tonight high", float(tonight["High"].max())), ("Tonight low", float(tonight["Low"].min()))]
    a = float(atr_series(df).iloc[-1])
    above = [(n, v) for n, v in lv if v > px]
    below = [(n, v) for n, v in lv if v < px]
    up = min(above, key=lambda x: x[1]) if above else None
    dn = max(below, key=lambda x: x[1]) if below else None
    return dict(price=px, atr=a, levels=[dict(name=n, value=v) for n, v in lv],
                above=dict(name=up[0], value=up[1], dist=up[1] - px) if up else None,
                below=dict(name=dn[0], value=dn[1], dist=px - dn[1]) if dn else None,
                bars_tonight=len(tonight), time=df.index[-1].isoformat())


# ---- phone alerts (ntfy) --------------------------------------------------------
def ntfy_send(topic, title, body, tags="chart_with_upwards_trend", priority="high"):
    """Push one notification to https://ntfy.sh/<topic>. The topic is the only 'password',
    so it comes from a GitHub secret (NTFY_TOPIC), never from the public repo."""
    req = urllib.request.Request(f"https://ntfy.sh/{topic}", data=body.encode("utf-8"), method="POST",
                                 headers={"Title": title, "Tags": tags, "Priority": priority,
                                          "Click": SITE_URL})
    with urllib.request.urlopen(req, timeout=15) as r:
        return r.status


def alert_text(c):
    dec = 1 if c["contract"] == "MGC" else 2
    f = lambda x: f"{x:,.{dec}f}"
    side = "LONG" if c["dir"] == 1 else "SHORT"
    return (f"{c['contract']} {c['tf']} {side} ({c['state']})\n"
            f"Entry {f(c['entry'])}  Stop {f(c['stop'])}  Target {f(c['target'])}\n"
            f"{c['r']:.1f}R  risk ${c['risk_micro']:,.0f}/{c['contract']} (${c['risk_full']:,.0f}/{c['full']})")


def send_alerts(new, now, slot):
    """One notification per scan that found NEW setups (already-logged boxes stay quiet)."""
    topic = os.environ.get("NTFY_TOPIC", "").strip()
    if not topic or not new:
        return None
    title = f"Zen futures: {len(new)} new setup{'s' if len(new) > 1 else ''} ({slot})"
    body = "\n\n".join(alert_text(c) for c in new[:6])
    try:
        return ntfy_send(topic, title, body)
    except Exception as e:
        print("alert failed:", e)
        return None


def send_test_alert(now):
    topic = os.environ.get("NTFY_TOPIC", "").strip()
    if not topic:
        print("NTFY_TOPIC secret is not set, so no alert was sent.")
        return None
    try:
        st = ntfy_send(topic, "Zen futures: test alert",
                       f"Alerts are working. Sent {now:%a %m/%d %I:%M %p} Central.\n"
                       "You'll get one like this when a scan finds a new MES, MNQ or MGC setup.",
                       tags="white_check_mark", priority="default")
        print("Test alert sent, status", st)
        return st
    except Exception as e:
        print("Test alert failed:", e)
        return None


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
                bar_time=df.index[-1].isoformat(),
                chart=dict(chart_bars(df), span=box_span(df, float(sig["top"]), float(sig["bot"]),
                                                          int(sig["age"]))))


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
                                bars=box["days"], to_roof=box["to_roof"], to_floor=box["to_floor"],
                                chart=dict(chart_bars(df), span=(box["days"] + 1, 1))))
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
    levels = {}
    for spec in CONTRACTS:
        try:
            lv = key_levels(frames["15m"].get(spec[2]))
            if lv:
                levels[spec[0]] = lv
        except Exception as e:
            print(f"{spec[0]} levels failed: {e}")
    LAST_RUN.clear()
    LAST_RUN.update(time=now.isoformat(), slot=slot, setups=setups, watches=watches,
                    levels=levels, contracts=[c[0] for c in CONTRACTS])
    new = log_setups(setups, now, slot)
    print_futures(setups, watches, now, slot)
    print_levels(levels)
    st = send_alerts(new, now, slot)
    if st:
        print(f"\nPhone alert sent for {len(new)} new setup(s).")
    return setups


def print_levels(levels):
    if not levels:
        return
    print("\nKey levels (15-min bars, Central):")
    for name, lv in levels.items():
        dec = 1 if name == "MGC" else 2
        parts = "  ".join(f"{l['name']} {l['value']:,.{dec}f}" for l in lv["levels"])
        print(f"  {name} now {lv['price']:,.{dec}f} | {parts}")


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
    rows, fresh = [], []
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
        fresh.append(c)
    if rows:
        new = pd.DataFrame(rows, columns=LOG_COLS)
        log = new if log.empty else pd.concat([log, new], ignore_index=True)
        log.to_csv(LOG_FILE, index=False)
    return fresh


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
