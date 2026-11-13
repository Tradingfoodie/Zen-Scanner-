#!/usr/bin/env python3
"""
Zen Quant Scanner
=================
Finds option setups before the move. Two styles, scored and logged separately:
  SWING: options 2-6 weeks out, held days to a couple of weeks (main focus)
  SHORT: options 3-10 days out, for comparison

Signals: unusual options flow, earnings before expiry, cheap relative to a
normal-sized move, momentum in the trade's direction, and volatility squeeze.

Setup (once):
    pip install yfinance pandas numpy

Run:
    python zen_scanner.py                       # both styles, default universe
    python zen_scanner.py --style swing         # swing only
    python zen_scanner.py --tickers AAOI V UNH  # specific tickers
    python zen_scanner.py --review              # grade logged picks

Every scan saves its picks to zen_log.csv so the review can grade wins AND losses.
Data: Yahoo Finance, free, ~15 min delayed. Research tool, not financial advice.
"""

import argparse
import datetime as dt
import logging
import math
import os
import sys
import time

try:
    import numpy as np
    import pandas as pd
    import yfinance as yf
except ImportError:
    sys.exit("Missing packages. Run:  pip install yfinance pandas numpy")

logging.getLogger("yfinance").setLevel(logging.CRITICAL)  # hide noisy ETF/earnings errors

LOG_FILE = "zen_log.csv"


def market_today():
    """Today's date in New York (Colab runs on UTC, which flips to tomorrow in the evening)."""
    try:
        from zoneinfo import ZoneInfo
        return dt.datetime.now(ZoneInfo("America/New_York")).date()
    except Exception:
        return dt.date.today()

DEFAULT_UNIVERSE = [
    # your recent winners
    "AAOI", "V", "UNH", "XLV", "CRWV", "LULU", "WMT", "DIS",
    # liquid names with active options
    "AAPL", "MSFT", "NVDA", "AMD", "AMZN", "META", "GOOGL", "TSLA", "NFLX",
    "AVGO", "ORCL", "CRM", "ADBE", "INTC", "MU", "SMCI", "PLTR", "COIN",
    "MSTR", "HOOD", "SOFI", "SHOP", "UBER", "ABNB", "SNOW", "NET", "CRWD",
    "PANW", "ARM", "DELL", "ANET", "JPM", "BAC", "GS", "MA", "PYPL",
    "XOM", "CVX", "LLY", "NVO", "MRNA", "PFE", "JNJ", "CVS", "COST",
    "TGT", "HD", "NKE", "SBUX", "MCD", "BA", "CAT", "DE", "GE", "F", "GM",
    "RIVN", "NIO", "BABA", "SPY", "QQQ", "IWM", "XLF", "XLE", "SMH",
    # lower-priced names, where cheap contracts sit closer to the money
    "AAL", "CCL", "SNAP", "LYFT", "MARA", "RIOT", "PLUG", "LCID", "T", "VZ",
]

# ---- Tunable settings per style ---------------------------------------------
STYLES = {
    "swing": dict(
        label="SWING", top=10,
        min_dte=14, max_dte=45,
        targets=(21, 35),        # check the expirations nearest 3 and 5 weeks out
        strike_band=0.12,        # strikes within +/-12% of the stock price
        max_premium=1.50,        # $150 per contract cap
        min_option_volume=20,
        flow_full=1.0,           # vol/OI that earns full flow points
        payoff_full=2.0,         # payoff multiple that earns full "cheap" points
    ),
    "short": dict(
        label="SHORT", top=5,
        min_dte=3, max_dte=10,
        targets=None,            # every expiration in range
        strike_band=0.05,
        max_premium=1.00,        # $100 per contract cap
        min_option_volume=50,
        flow_full=2.0,
        payoff_full=3.0,
    ),
    # LOTTERY LIST (test from 11/16): cheap contracts 1-7 days out, graded on whether
    # they DOUBLE at any point (sell into the pop), not on where they close.
    "lotto": dict(
        label="LOTTERY", top=5,
        min_dte=1, max_dte=7,
        targets=None,
        strike_band=0.04,
        max_premium=0.50,        # $50 per contract cap
        min_option_volume=100,
        flow_full=2.0,
        payoff_full=3.0,
    ),
}
MAX_SPREAD_PCT = 0.20  # skip contracts with (ask - bid) / mid above 20%
REQUIRE_BOX = False    # True = only list/log picks with a clean box breakout/retest (B tag)
SKIP_WARNINGS = True   # never list picks carrying a WARNING (! tag)
BREAK_MARGIN_ATR = 0.15  # Box Watch: price must clear the edge by this many ATRs to count as breaking
SCANNER_VERSION = "v31"  # written to the log so the review can tell versions apart

WEIGHTS = {  # max points per signal; total = 100
    "unusual_flow": 25,   # near-the-money volume vs open interest
    "catalyst": 15,       # earnings before the option expires
    "cheap_ntm": 20,      # payoff if the stock makes a normal-sized move
    "momentum": 15,       # trend + relative volume in the trade's direction
    "squeeze": 10,        # Bollinger width near its 6-month low
    "box": 15,            # broke out of a consolidation box and is retesting the edge
}


def nz(x):
    try:
        x = float(x)
        return 0.0 if math.isnan(x) else x
    except (TypeError, ValueError):
        return 0.0


# ---- Signals ----------------------------------------------------------------
def stock_signals(hist):
    c, h, l, v = hist["Close"], hist["High"], hist["Low"], hist["Volume"]
    price = float(c.iloc[-1])
    sma20, sma50 = c.rolling(20).mean(), c.rolling(50).mean()
    std20 = c.rolling(20).std()
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = float(tr.rolling(14).mean().iloc[-1])
    # Today's bar is partial during market hours, so use the larger of today/yesterday.
    base = v.iloc[-22:-2].mean()
    rel_vol = float(max(v.iloc[-1], v.iloc[-2]) / base) if base > 0 else 0.0
    ret5 = float(c.iloc[-1] / c.iloc[-6] - 1)
    ret20 = float(c.iloc[-1] / c.iloc[-21] - 1)
    bbw = (4 * std20 / sma20).dropna().iloc[-120:]
    squeeze_pct = float((bbw < bbw.iloc[-1]).mean())  # 0 = tightest in 6 months

    def trend(ma, ret):
        above = price > ma
        return 1 if (above and ret > 0) else -1 if (not above and ret < 0) else 0

    return dict(price=price, atr=atr, rel_vol=rel_vol, ret5=ret5, ret20=ret20,
                squeeze_pct=squeeze_pct,
                trend_short=trend(sma20.iloc[-1], ret5),
                trend_swing=trend(sma50.iloc[-1], ret20))


MIN_TOUCHES = 2   # Potter: a box needs at least 2 touches on the roof AND 2 on the floor


def touches(near, gap=3):
    """Count separate tests of an edge: runs of 'near the edge' bars split by >= gap bars away."""
    count, last = 0, -10**9
    for i, hit in enumerate(near):
        if hit:
            if i - last > gap:
                count += 1
            last = i
    return count


def has_touches(btop, bbot, top, bot, a):
    """True if the roof and the floor were each tested at least MIN_TOUCHES times."""
    r = top - bot
    tol = min(max(0.15 * r, 0.25 * a), 0.3 * r)
    roof = touches(list(btop >= top - tol))
    floor = touches(list(bbot <= bot + tol))
    return roof >= MIN_TOUCHES and floor >= MIN_TOUCHES


def find_old_boxes(body_top, body_bot, c, atr, skip_last, lookback=250, step=5):
    """Older Potter boxes anywhere in the past year (tight, flat, 2+ touches on each edge),
    excluding the most recent `skip_last` bars. Used to find the NEXT box a breakout heads to."""
    boxes = []
    n = len(c)
    first = max(20, n - lookback)
    for j in range(n - skip_last, first, -step):   # window ends at j (exclusive)
        for L in (40, 30, 20, 15, 10):
            if j - L < first - 20:
                continue
            bt, bb = body_top.iloc[j - L:j], body_bot.iloc[j - L:j]
            top, bot = float(bt.max()), float(bb.min())
            a = float(atr.iloc[j - 1])
            r = top - bot
            if not a > 0 or r <= 0 or r > 0.7 * math.sqrt(L) * a:
                continue
            win = c.iloc[j - L:j]
            third = max(L // 3, 2)
            if abs(win.iloc[-third:].mean() - win.iloc[:third].mean()) > 0.35 * r:
                continue
            if not has_touches(bt, bb, top, bot, a):
                continue
            boxes.append((top, bot))
            break                       # longest valid box ending here is enough
    return boxes


def next_box_target(boxes, d, edge, a):
    """Potter trades box to box: the target is the near edge of the nearest older box
    beyond the broken edge (with at least half a day's range of empty space between)."""
    if d == 1:
        cands = [(bot, top) for top, bot in boxes if bot >= edge + 0.5 * a]
        return min(cands) if cands else None          # (near edge = its floor, its roof)
    cands = [(top, bot) for top, bot in boxes if top <= edge - 0.5 * a]
    return max(cands) if cands else None              # (near edge = its roof, its floor)


def box_signal(hist):
    """Potter-style box: roof and floor drawn on candle BODIES (wicks may poke through),
    10-40 days long and tight vs ATR. Finds the most recent box that price broke out of
    in the last 3-15 days and says whether price is now retesting the broken edge.
    Cost basis = the box's 50% mark. Target = the near edge of the NEXT older box beyond
    the break (box to box); if none, the box height projected past the edge."""
    c, h, l, o = hist["Close"], hist["High"], hist["Low"], hist["Open"]
    body_top, body_bot = np.maximum(o, c), np.minimum(o, c)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    px, a_now = float(c.iloc[-1]), float(atr.iloc[-1])
    if not a_now > 0:
        return None
    for e in range(3, 16):              # box ended e bars ago (most recent first)
        for L in (40, 30, 20, 15, 10):  # box length in bars (longest first)
            if e + L + 15 > len(c):
                continue
            top = float(body_top.iloc[-(e + L):-e].max())
            bot = float(body_bot.iloc[-(e + L):-e].min())
            a = float(atr.iloc[-e - 1])
            if not a > 0 or (top - bot) > 0.7 * math.sqrt(L) * a:
                continue                # not tight enough to be a box
            if bot <= c.iloc[-e] <= top:
                continue                # the breakout must start the day after the box ends
            r, last = top - bot, float(c.iloc[-e - 1])
            win = c.iloc[-(e + L):-e]
            third = max(L // 3, 2)
            if abs(win.iloc[-third:].mean() - win.iloc[:third].mean()) > 0.35 * r:
                continue                # drifting, not sideways: not a box
            if not has_touches(body_top.iloc[-(e + L):-e], body_bot.iloc[-(e + L):-e], top, bot, a):
                continue                # needs 2+ touches on the roof and the floor
            if not (bot + 0.1 * r <= last <= top - 0.1 * r):
                continue                # box's last close sits at an edge: window ate the breakout
            after = c.iloc[-e:]
            m = 0.5 * a                 # breakout must clear the edge decisively
            up, dn = bool((after > top + m).any()), bool((after < bot - m).any())
            if up == dn:
                continue                # no breakout yet, or broke both ways
            height = top - bot
            if up:
                d, edge, target, dist = 1, top, top + height, (px - top) / a_now
            else:
                d, edge, target, dist = -1, bot, bot - height, (bot - px) / a_now
            # Box to box: aim for the next older box beyond the break, if there is one.
            nb = next_box_target(find_old_boxes(body_top, body_bot, c, atr, e + L), d, edge, a)
            kind, nxt = "measured", None
            if nb:
                target, kind = nb[0], "next box"
                nxt = (min(nb), max(nb))
            state = "failed" if dist < -0.5 else "retest" if dist <= 1.0 else "extended"
            return dict(dir=d, top=top, bot=bot, mid=(top + bot) / 2,
                        target=target, kind=kind, next_box=nxt, state=state, age=e)
    return None


def current_box(hist):
    """Box Watch: the most recent tight sideways box, drawn from the days BEFORE today
    (so a breakout today can't stretch the box to swallow itself). Same Potter rules:
    candle bodies, 10-40 days, tight vs ATR, flat. Status says whether today's price is
    still inside, or is breaking out of it right now (within 1.5 ATR of the edge)."""
    c, h, l, o = hist["Close"], hist["High"], hist["Low"], hist["Open"]
    body_top, body_bot = np.maximum(o, c), np.minimum(o, c)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    a = float(tr.rolling(14).mean().iloc[-1])
    px = float(c.iloc[-1])
    if not a > 0:
        return None
    best, base_r = None, None
    for L in (10, 15, 20, 30, 40):      # start short, stretch while the box stays the same size
        if L + 16 > len(c):
            break
        top = float(body_top.iloc[-(L + 1):-1].max())
        bot = float(body_bot.iloc[-(L + 1):-1].min())
        r = top - bot
        if r <= 0 or r > 0.7 * math.sqrt(L) * a:
            continue
        win = c.iloc[-(L + 1):-1]
        third = max(L // 3, 2)
        if abs(win.iloc[-third:].mean() - win.iloc[:third].mean()) > 0.35 * r:
            continue
        if not has_touches(body_top.iloc[-(L + 1):-1], body_bot.iloc[-(L + 1):-1], top, bot, a):
            continue                    # needs 2+ touches on the roof and the floor
        if not (bot - 1.5 * a <= px <= top + 1.5 * a):
            continue                    # too far outside: old news, not a box to watch
        if base_r is None:
            base_r = r
        elif r > 1.25 * base_r:
            break                       # stretching further pulls in non-box candles
        m = BREAK_MARGIN_ATR * a        # a few cents over the edge is a test, not a break
        status = "up" if px > top + m else "down" if px < bot - m else "inside"
        best = dict(top=top, bot=bot, mid=(top + bot) / 2, days=L, price=px, status=status,
                    to_roof=max(top - px, 0) / a, to_floor=max(px - bot, 0) / a)
    return best


def next_earnings(tk, today):
    try:
        cal = tk.calendar
        dates = []
        if isinstance(cal, dict):
            dates = cal.get("Earnings Date", []) or []
        elif cal is not None and hasattr(cal, "index") and "Earnings Date" in cal.index:
            dates = list(cal.loc["Earnings Date"].values)
        dates = [pd.Timestamp(d).date() for d in dates]
        future = [d for d in dates if d >= today]
        return min(future) if future else None
    except Exception:
        return None


def pick_expiries(all_exps, today, st):
    cands = []
    for exp in all_exps:
        d = dt.date.fromisoformat(exp)
        dte = (d - today).days
        if st["min_dte"] <= dte <= st["max_dte"]:
            cands.append((exp, dte))
    if not cands or not st["targets"]:
        return [e for e, _ in cands]
    chosen = {min(cands, key=lambda x: abs(x[1] - t))[0] for t in st["targets"]}
    return sorted(chosen)


def option_rows(tk, price, today, exps, band):
    rows = []
    for exp in exps:
        exp_d = dt.date.fromisoformat(exp)
        dte = (exp_d - today).days
        chain = tk.option_chain(exp)
        for side, df in (("call", chain.calls), ("put", chain.puts)):
            df = df[(df["strike"] - price).abs() / price <= band]
            for _, r in df.iterrows():
                bid, ask = nz(r.get("bid")), nz(r.get("ask"))
                if not (bid > 0 and ask >= bid):
                    continue            # no two-sided quote: a stale last price is not a real cost
                mid = (bid + ask) / 2
                spread = (ask - bid) / mid
                vol, oi = nz(r.get("volume")), nz(r.get("openInterest"))
                rows.append(dict(side=side, expiry=exp_d, dte=dte, strike=float(r["strike"]),
                                 mid=mid, spread_pct=spread, volume=vol, oi=oi,
                                 vol_oi=vol / max(oi, 1), iv=nz(r.get("impliedVolatility")),
                                 contract=r.get("contractSymbol", "")))
    return pd.DataFrame(rows)


def score_side(side, o, s, earn, style, box=None, today=None):
    st = STYLES[style]
    flow = o["volume"].sum() / max(o["oi"].sum(), 1)
    liq = o[(o["volume"] >= st["min_option_volume"]) & (o["mid"] > 0)
            & (o["mid"] <= st["max_premium"]) & (o["spread_pct"] <= MAX_SPREAD_PCT)]
    if liq.empty:
        return None
    liq = liq.assign(dist=(liq["strike"] - s["price"]).abs())
    # If the chart points this way, prefer strikes the box target can actually pay off:
    # the option's value at the target must at least cover what you paid for it.
    want = 1 if side == "call" else -1
    aligned = bool(box and box["dir"] == want and box["state"] in ("retest", "extended"))
    reach_note = ""
    past = False
    if aligned:
        tgt = box["target"]
        past = (s["price"] >= tgt) if side == "call" else (s["price"] <= tgt)
    if aligned and not past:
        at_tgt = (tgt - liq["strike"]).clip(lower=0) if side == "call" else (liq["strike"] - tgt).clip(lower=0)
        liq = liq.assign(tgt_mult=at_tgt / liq["mid"])
        reach = liq[liq["tgt_mult"] >= 1.0]
        if not reach.empty:
            liq = reach
        else:
            reach_note = "target short of strike"
    pick = liq.sort_values(["vol_oi", "dist"], ascending=[False, True]).iloc[0]

    pts, why = {}, []
    # 1. Unusual flow
    pts["unusual_flow"] = WEIGHTS["unusual_flow"] * min(flow / st["flow_full"], 1)
    why.append(f"flow {flow:.1f}x OI")
    # 2. Catalyst: earnings on or before expiry
    if earn and today and earn == today:
        pts["catalyst"] = WEIGHTS["catalyst"] * 0.5   # may already be out (before the open)
        why.append("earnings TODAY (may be out)")
    elif earn and earn <= pick["expiry"]:
        pts["catalyst"] = WEIGHTS["catalyst"]
        why.append(f"earnings {earn:%m/%d}")
    else:
        pts["catalyst"] = 0
    # 3. Cheap vs. a normal move: payoff if stock moves 1 expected move (ATR * sqrt(days))
    em = s["atr"] * math.sqrt(max(pick["dte"], 1))
    target = s["price"] + em if side == "call" else s["price"] - em
    payoff = max(0, target - pick["strike"]) if side == "call" else max(0, pick["strike"] - target)
    ratio = payoff / pick["mid"] if pick["mid"] > 0 else 0
    pts["cheap_ntm"] = WEIGHTS["cheap_ntm"] * min(ratio / st["payoff_full"], 1)
    why.append(f"{ratio:.1f}x on normal move")
    # 4. Momentum in the trade's direction (5-day for short, 20-day vs 50MA for swing)
    tr = s["trend_swing"] if style == "swing" else s["trend_short"]
    align = 1.0 if tr == want else 0.3 if tr == 0 else 0.0
    pts["momentum"] = WEIGHTS["momentum"] * align * min(s["rel_vol"] / 2, 1)
    ret = f"20d {s['ret20']:+.1%}" if style == "swing" else f"5d {s['ret5']:+.1%}"
    why.append(f"{ret}, RVOL {s['rel_vol']:.1f}x")
    # 5. Squeeze
    pts["squeeze"] = WEIGHTS["squeeze"] * max(0, (0.5 - s["squeeze_pct"]) / 0.5)
    if s["squeeze_pct"] < 0.25:
        why.append("squeeze")
    # 6. Box breakout in the trade's direction: retest = full points, extended = partial
    box_state = ""
    if aligned:
        box_state = box["state"]
        pts["box"] = WEIGHTS["box"] * (1.0 if box_state == "retest" else 0.4)
        why.append(f"box {box_state} {box['bot']:.2f}-{box['top']:.2f}")
        if box.get("next_box"):
            nb_lo, nb_hi = box["next_box"]
            why.append(f"next box {nb_lo:.2f}-{nb_hi:.2f}")
            why.append(f"mid ${box['mid']:.2f} | tgt ${box['target']:.2f}")
        else:
            why.append(f"mid ${box['mid']:.2f} | tgt ${box['target']:.2f} (measured)")
        if past:
            pts["box"] = 0              # the move to the target already happened
            why.append("WARNING: already past target")
        elif reach_note:
            pts["box"] *= 0.5           # chart agrees, but no affordable strike reaches the target
            why.append(f"WARNING: {reach_note}")
        else:
            why.append(f"worth {pick['tgt_mult']:.1f}x cost at target")
    elif box and box["dir"] == -want and box["state"] in ("retest", "extended"):
        box_state = "against"   # fresh breakout the OTHER way: this pick fights the chart
        pts["box"] = -WEIGHTS["box"]
        way = "up" if box["dir"] == 1 else "down"
        why.append(f"WARNING: box broke {way}")
    else:
        pts["box"] = 0

    return dict(score=round(sum(pts.values()), 1), side=side, strike=pick["strike"],
                box_state=box_state,
                box_top=round(box["top"], 2) if box_state else None,
                box_bot=round(box["bot"], 2) if box_state else None,
                box_mid=round(box["mid"], 2) if box_state else None,
                box_target=round(box["target"], 2) if box_state else None,
                expiry=pick["expiry"], dte=int(pick["dte"]), mid=round(pick["mid"], 2),
                opt_vol=int(pick["volume"]), vol_oi=round(pick["vol_oi"], 1),
                iv=round(pick["iv"], 2), contract=pick["contract"], why="; ".join(why),
                **{f"pts_{k}": round(v, 1) for k, v in pts.items()})


def scan_ticker(sym, today, styles):
    tk = yf.Ticker(sym)
    hist = tk.history(period="1y", interval="1d", auto_adjust=False)
    if len(hist) < 60:
        return [], None
    s = stock_signals(hist)
    box = box_signal(hist)
    earn = next_earnings(tk, today)
    cur = current_box(hist)
    watch = dict(ticker=sym, earn=earn, **cur) if cur else None
    all_exps = tk.options
    out = []
    for style in styles:
        exps = pick_expiries(all_exps, today, STYLES[style])
        if not exps:
            continue
        opts = option_rows(tk, s["price"], today, exps, STYLES[style]["strike_band"])
        if opts.empty:
            continue
        sides = [score_side(side, opts[opts["side"] == side], s, earn, style, box, today)
                 for side in ("call", "put")]
        sides = [x for x in sides if x]
        if sides:
            best = max(sides, key=lambda x: x["score"])
            out.append(dict(style=style, ticker=sym, price=round(s["price"], 2), **best))
    return out, watch


# ---- Commands ---------------------------------------------------------------
def print_card(n, r):
    tag = "STRONG" if r["score"] >= 70 else "watch" if r["score"] >= 50 else "weak"
    print(f"#{n} {r['ticker']} {r['side'].upper()} {r['strike']:g}  {r['score']:.0f} {tag}")
    print(f"   exp {pd.Timestamp(r['expiry']):%m/%d} ({r['dte']}d) | ${r['mid'] * 100:,.0f}/ct")
    print(f"   stock ${r['price']:,.2f} | opt vol {r['opt_vol']:,}")
    for reason in str(r["why"]).split("; "):
        print(f"   - {reason}")
    print()


def print_box_watch(watches, top=10):
    if not watches:
        return
    # Breaking boxes first; then longer boxes with price closer to an edge.
    for w in watches:
        w["rank"] = w["days"] / (1 + min(w["to_roof"], w["to_floor"]))
        if w["status"] != "inside":
            w["rank"] += 1000
    watches = sorted(watches, key=lambda w: w["rank"], reverse=True)[:top]
    print("===== BOX WATCH =====")
    print("Boxes to watch. Set the two alerts")
    print("shown in TradingView:\n")
    for w in watches:
        print(f"{w['ticker']}  box {w['bot']:.2f}-{w['top']:.2f} ({w['days']}d)")
        print(f"   mid ${w['mid']:.2f} | now ${w['price']:.2f}")
        if w["status"] == "down":
            # Breaking down today: wait for the punchback, wrong if it reclaims the mid.
            print(f"   BREAKING DOWN below floor {w['bot']:.2f}")
            print(f"   \u2193 punchback to {w['bot']:.2f} that fails \u2192 puts")
            print(f"   \u2191 back above {w['mid']:.2f} (mid) = failed break")
        elif w["status"] == "up":
            print(f"   BREAKING UP above roof {w['top']:.2f}")
            print(f"   \u2191 punchback to {w['top']:.2f} that holds \u2192 calls")
            print(f"   \u2193 back below {w['mid']:.2f} (mid) = failed break")
        elif w["price"] < w["mid"]:
            # Below the cost basis: a reclaim of the mid is the call trigger,
            # losing the floor is the put trigger.
            print("   below mid, near floor")
            print(f"   \u2191 above {w['mid']:.2f} (mid) \u2192 calls")
            print(f"   \u2193 below {w['bot']:.2f} (floor) \u2192 puts")
        else:
            # Above the cost basis: losing the mid is the put trigger,
            # breaking the roof is the call trigger.
            print("   above mid, near roof")
            print(f"   \u2193 below {w['mid']:.2f} (mid) \u2192 puts")
            print(f"   \u2191 above {w['top']:.2f} (roof) \u2192 calls")
        if w["earn"]:
            print(f"   earnings {w['earn']:%m/%d}")
        print()


SHOW_DETAILS = True   # False = only the short summary at the top


def pick_tags(r):
    """One-letter tags for the summary: B box retest, E earnings, F strong flow, ! warning."""
    why = str(r.get("why", ""))
    t = ""
    if r.get("box_state") in ("retest", "extended") and "WARNING" not in why:
        t += "B"
    if "earnings" in why:
        t += "E"
    if r.get("pts_unusual_flow", 0) >= 0.6 * WEIGHTS["unusual_flow"]:
        t += "F"
    if "WARNING" in why:
        t += "!"
    return t


def print_summary(picks, watches, today, top=8):
    """Phone-sized digest: the best picks of the day on one line each, then Box Watch."""
    best = picks.sort_values("score", ascending=False).head(top)
    print(f"\n===== TODAY'S PICKS {today:%a %m/%d} =====")
    print("#  TICKER  STRIKE   EXP   COST SC")
    for i, (_, r) in enumerate(best.iterrows(), 1):
        side = "C" if r["side"] == "call" else "P"
        sty = {"short": "s", "lotto": "L"}.get(r["style"], "w")
        cost = f"${r['mid'] * 100:,.0f}".rjust(5)
        print(f"{i:<2} {r['ticker']:<5} {side} {r['strike']:<6g} {pd.Timestamp(r['expiry']):%m/%d} "
              f"{cost} {r['score']:>2.0f}{sty} {pick_tags(r)}")
    print("s=short w=swing L=lottery | B=box E=earnings")
    print("F=strong flow | !=warning, skip")
    breaking = [w for w in watches if w.get("status") in ("up", "down")]
    if breaking:
        print("\nBREAKING BOXES (wait for punchback)")
        for w in breaking:
            arrow = "\u2191" if w["status"] == "up" else "\u2193"
            edge = w["top"] if w["status"] == "up" else w["bot"]
            print(f"   {w['ticker']:<5} {arrow} edge {edge:.2f} | now {w['price']:.2f}")
    inside = [w for w in watches if w.get("status") == "inside"]
    if inside:
        print("\nIN A BOX (set two alerts)")
        for w in inside[:5]:
            hi = w["mid"] if w["price"] < w["mid"] else w["top"]
            lo = w["bot"] if w["price"] < w["mid"] else w["mid"]
            print(f"   {w['ticker']:<5} \u2191{hi:.2f} \u2193{lo:.2f}")
    print()


# ---- Sector test (11/9-11/20): MEASUREMENT ONLY, never changes scores or picks ----
# Same groups as the 11/6 tally, so the forward test checks the same idea.
DEFENSIVE = {"WMT", "COST", "TGT", "MCD", "PFE", "CVS", "NVO", "JNJ", "UNH", "LLY",
             "MRNA", "VZ", "T", "XLV"}
TECH = {"AAPL", "MSFT", "NVDA", "AMD", "GOOGL", "META", "AMZN", "CRM", "ADBE", "ORCL",
        "INTC", "MU", "SMCI", "PLTR", "CRWV", "NFLX", "TSLA", "AVGO", "ARM", "NET", "SNOW",
        "CRWD", "PANW", "DELL", "ANET", "SHOP", "HOOD", "COIN", "MSTR", "AAOI", "SMH", "QQQ"}
SECTOR_ETF = {  # sector fund used for "is money flowing into this sector?"
    **{t: "XLK" for t in ("AAPL", "MSFT", "NVDA", "AMD", "AVGO", "ORCL", "CRM", "ADBE", "INTC",
                          "MU", "SMCI", "PLTR", "NET", "SNOW", "CRWD", "PANW", "ARM", "DELL",
                          "ANET", "CRWV", "AAOI", "SMH", "SHOP")},
    **{t: "XLC" for t in ("META", "GOOGL", "NFLX", "DIS", "SNAP", "T", "VZ")},
    **{t: "XLY" for t in ("AMZN", "TSLA", "NKE", "SBUX", "MCD", "HD", "ABNB", "UBER", "LYFT",
                          "RIVN", "LCID", "F", "GM", "NIO", "BABA", "LULU", "CCL")},
    **{t: "XLP" for t in ("WMT", "COST", "TGT")},
    **{t: "XLV" for t in ("UNH", "LLY", "NVO", "MRNA", "PFE", "JNJ", "CVS", "XLV")},
    **{t: "XLF" for t in ("JPM", "BAC", "GS", "MA", "V", "PYPL", "SOFI", "HOOD", "COIN", "XLF")},
    **{t: "XLE" for t in ("XOM", "CVX", "XLE")},
    **{t: "XLI" for t in ("BA", "CAT", "DE", "GE", "AAL", "PLUG")},
}


def sector_strength():
    """5-day return of each sector fund minus SPY's (+ = money flowing in)."""
    out = {}
    try:
        funds = sorted(set(SECTOR_ETF.values()))
        data = yf.download(funds + ["SPY"], period="1mo", interval="1d", progress=False,
                           auto_adjust=False, group_by="ticker", threads=True)

        def r5(t):
            c = data[t]["Close"].dropna()
            return float(c.iloc[-1] / c.iloc[-6] - 1)
        spy = r5("SPY")
        for f in funds:
            try:
                out[f] = r5(f) - spy
            except Exception:
                pass
    except Exception:
        pass
    return out


def tag_sectors(df, rs):
    """Add sector columns to the picks. Measurement only."""
    df = df.copy()
    df["group"] = df["ticker"].map(lambda t: "defensive" if t in DEFENSIVE
                                   else "tech" if t in TECH else "other")
    df["sector"] = df["ticker"].map(lambda t: SECTOR_ETF.get(t, ""))
    df["sector_rs5"] = df["sector"].map(lambda f: round(rs[f], 4) if f in rs else None)

    def with_flow(r):
        x = r["sector_rs5"]
        if x is None or pd.isna(x):
            return ""
        return "with" if (x > 0) == (r["side"] == "call") else "against"
    df["with_sector"] = df.apply(with_flow, axis=1)
    return df


SESSION = "am"     # "am" = midday scan (~11:30 CT), "pm" = evening scan (~2:15 CT, before the close)
LAST_SCAN = {}     # structured results of the latest scan (used by the website)
LAST_REVIEW = None  # graded picks from the latest review (used by the website)


def run_scan(tickers, today, styles=("swing", "short", "lotto")):
    if today.weekday() >= 5:
        print("Market is closed on weekends, so no scan was run")
        print("or logged. Run the review cell instead.")
        return
    results, skipped, watches = [], [], []
    try:
        from zoneinfo import ZoneInfo
        now = dt.datetime.now(ZoneInfo("America/New_York")).time()
        if not (dt.time(9, 45) <= now <= dt.time(16, 0)):
            print("NOTE: market is not open. Option costs are last")
            print("session's quotes and may be missing or wide at")
            print("the open. Re-check prices before acting.\n")
    except Exception:
        pass
    print(f"Scanning {len(tickers)} tickers... (3-6 min)", flush=True)
    for i, sym in enumerate(tickers, 1):
        try:
            picks, watch = scan_ticker(sym, today, styles)
            results.extend(picks)
            if watch:
                watches.append(watch)
        except Exception:  # one bad ticker shouldn't kill the scan
            skipped.append(sym)
        if i % 10 == 0:
            print(f"  {i}/{len(tickers)} done", flush=True)
        time.sleep(0.25)
    if skipped:
        print(f"  skipped (no data): {', '.join(skipped)}")
    LAST_SCAN.clear()
    LAST_SCAN.update(watches=watches, picks=None, rs={})
    if not results:
        print("No setups found. Markets may be closed or Yahoo is throttling; try again later.")
        return
    all_df = pd.DataFrame(results)
    tags = all_df.apply(pick_tags, axis=1)
    n0 = len(all_df)
    if SKIP_WARNINGS:
        all_df = all_df[~tags.str.contains("!", regex=False)]
    if REQUIRE_BOX:
        all_df = all_df[tags.loc[all_df.index].str.contains("B", regex=False)]
    if all_df.empty:
        print(f"No picks passed the box/warning filters ({n0} scored).")
        print("Nothing logged. Use the box lists below for alerts.")
        print_summary(pd.DataFrame(columns=["score"]), watches, today)
        return
    keep = []
    for style in styles:
        st = STYLES[style]
        df = all_df[all_df["style"] == style].sort_values("score", ascending=False).head(st["top"])
        if not df.empty:
            keep.append(df)
    if not keep:
        print("No setups found today.")
        return
    print_summary(pd.concat(keep, ignore_index=True), watches, today)
    LAST_SCAN["picks"] = pd.concat(keep, ignore_index=True)
    if SHOW_DETAILS:
        for df in keep:
            st = STYLES[df["style"].iloc[0]]
            print(f"\n===== {st['label']} {today:%a %m/%d} =====")
            print(f"{st['min_dte']}-{st['max_dte']} days out | top {len(df)}")
            print("70+ strong | 50-70 watch\n")
            for n, (_, r) in enumerate(df.iterrows(), 1):
                print_card(n, r)
        print_box_watch(watches)
    else:
        print("(Full cards hidden. Set SHOW_DETAILS = True")
        print(" in the settings to see them.)")
    new = pd.concat(keep, ignore_index=True)
    new.insert(0, "scan_date", today)
    new["ver"] = SCANNER_VERSION
    new["session"] = SESSION
    rs = {}
    try:
        rs = sector_strength()
        new = tag_sectors(new, rs)
        LAST_SCAN["rs"] = rs
        LAST_SCAN["picks"] = tag_sectors(LAST_SCAN["picks"], rs)
    except Exception:
        pass                            # the sector tag is a bonus; never block the log
    if os.path.exists(LOG_FILE):
        new = pd.concat([pd.read_csv(LOG_FILE), new], ignore_index=True)
    new.to_csv(LOG_FILE, index=False)
    print("Saved to your pick log.")
    if rs:
        names = {"XLK": "tech", "XLC": "comm", "XLY": "discr", "XLP": "staples",
                 "XLV": "health", "XLF": "fin", "XLE": "energy", "XLI": "indust"}
        order = sorted(rs.items(), key=lambda kv: kv[1], reverse=True)
        print("\nSECTOR FLOW (5 days vs SPY)")
        for f, x in order:
            print(f"   {names.get(f, f):<8} {x:+.1%}")


def live_option_price(tk, chains, r, exp_d):
    """Current mid (or last) price of a logged contract, or None if unavailable."""
    try:
        key = exp_d.isoformat()
        if key not in chains:
            chains[key] = tk.option_chain(key)
        ch = chains[key]
        df = ch.calls if r["side"] == "call" else ch.puts
        c = r.get("contract")
        row = df[df["contractSymbol"] == c] if isinstance(c, str) and c else df.iloc[0:0]
        if row.empty:
            row = df[(df["strike"] - float(r["strike"])).abs() < 1e-6]
        if row.empty:
            return None
        x = row.iloc[0]
        bid, ask = nz(x.get("bid")), nz(x.get("ask"))
        p = (bid + ask) / 2 if (bid > 0 and ask > 0) else nz(x.get("lastPrice"))
        return p if p > 0 else None
    except Exception:
        return None


WIN_AT, CUT_AT = 2.0, 0.5   # exit rules: sell at +100% (2x entry), cut at -50% (0.5x entry)
CUT_WIDE = 0.3              # comparison: a looser -70% stop


def option_hist(contract, scan_d, today):
    """Daily price history of the contract itself since the scan, if Yahoo has it."""
    if not isinstance(contract, str) or not contract:
        return None
    try:
        h = yf.Ticker(contract).history(start=scan_d + dt.timedelta(days=1),
                                       end=today + dt.timedelta(days=1), auto_adjust=False)
        return h if not h.empty else None
    except Exception:
        return None


def exit_grade(h, entry, final_ret, peak_ret, expired, cut=CUT_AT):
    """Grade a pick the way you'd trade it: WIN if it hit +100% first, LOSS if it hit
    -50% first, else open (or, once expired, its final result). Uses the contract's own
    daily highs/lows when available; on a day that touched both lines, counts the loss."""
    if entry > 0 and h is not None:
        for d, row in h.iterrows():
            if nz(row.get("Low")) > 0 and row["Low"] <= entry * cut:
                return "LOSS", pd.Timestamp(d).date(), False
            if row["High"] >= entry * WIN_AT:
                return "WIN", pd.Timestamp(d).date(), False
    # No contract history: estimate from best/current values.
    if peak_ret >= WIN_AT - 1:
        return "WIN", None, True
    if final_ret <= cut - 1:
        return "LOSS", None, True
    return ("EXPIRED" if expired else "open"), None, h is None


def split_exit(h, entry, grade, gdate, final_ret):
    """Return (as a multiple of cost) under the split exit: sell half at +100%; the other
    half rides and is sold at entry if price falls back there, else valued at its final
    (live or expiry) price. Losses are cut at -50% as usual. None if still undecided."""
    if grade == "LOSS":
        return CUT_AT - 1
    if grade == "EXPIRED":
        return final_ret
    if grade != "WIN":
        return None
    runner = max(final_ret, 0.0)            # no history: assume the break-even stop held
    if h is not None and gdate is not None:
        runner = final_ret
        for d, row in h.iterrows():
            if pd.Timestamp(d).date() <= gdate:
                continue
            if nz(row.get("Low")) > 0 and row["Low"] <= entry:
                runner = 0.0                # stopped out at break-even
                break
    return 0.5 * (WIN_AT - 1) + 0.5 * runner


def print_top_runners(df, n=10):
    """The picks that ran the most (by peak), so the home runs don't hide in the list."""
    top = df.sort_values("peak_ret", ascending=False).head(n)
    top = top[top["peak_ret"] > 0]
    if top.empty:
        return
    print(f"\n===== TOP {len(top)} RUNNERS =====")
    print("biggest peaks across all picks\n")
    for i, (_, r) in enumerate(top.iterrows(), 1):
        days = (r["gdate"] - r["scan"]).days if r["gdate"] is not None and r["grade"] == "WIN" else None
        print(f"#{i} {r['ticker']} {r['side'].upper()} {r['strike']:g}  exp {r['expiry']:%m/%d}")
        print(f"   {r['style']} | scanned {r['scan']:%m/%d} | score {r['score']:.0f}")
        print(f"   paid ${r['entry'] * 100:,.0f} | best {r['peak_ret']:+.0%} | now {r['final_ret']:+.0%}")
        tags = []
        if r["box"] and r["box"] != "against":
            tags.append(f"box {r['box']}")
        if r["earn"]:
            tags.append("earnings")
        if days is not None:
            tags.append(f"+100% in {days}d")
        if tags:
            print(f"   {' | '.join(tags)}")
        print()
    cheap = top[top["entry"] < 0.50]
    print(f"   cost under $50: {len(cheap)} of {len(top)}")
    print(f"   short list: {(top['style'] == 'short').sum()} of {len(top)}")


def rule_stats(dec, col, cut):
    """Wins, losses, decided count and $ result at $100 per trade for one exit rule."""
    w = int((dec[col] == "WIN").sum())
    l = int((dec[col] == "LOSS").sum())
    mid = dec[dec[col] == "EXPIRED"]
    dollars = w * 100 * (WIN_AT - 1) + l * 100 * (cut - 1) + float((mid["final_ret"] * 100).sum())
    return w, l, len(dec), dollars


def print_sector_test(df):
    """Sector test (11/9-11/20): does side-of-the-rotation matter going forward?"""
    d = df[df["grade"].isin(["WIN", "LOSS"])].copy()
    d = d[d["group"].fillna("").astype(str) != ""]
    if d.empty:
        return
    print("\n===== SECTOR TEST (since 11/9) =====")
    print("(+100% / -50% win rate; measurement only)")
    for (grp, side), g in d.groupby(["group", "side"]):
        w = int((g["grade"] == "WIN").sum())
        print(f"   {grp} {side}s: {w} of {len(g)} ({w / len(g):.0%})")
    ws = d[d["with_sector"].fillna("").astype(str) != ""]
    for lab, g in ws.groupby("with_sector"):
        w = int((g["grade"] == "WIN").sum())
        print(f"   {lab} sector flow: {w} of {len(g)} ({w / len(g):.0%})")
    print("   11/6 tally to beat: defensive calls 67%,")
    print("   tech calls 18%. Under ~20 picks a group = too early.")


def print_overnight(df):
    """Potter's idea: buy near the close, sell into the next morning's move.
    Compares evening (pm) picks with midday (am) picks on the first trading day after the scan."""
    if "session" not in df.columns:
        return
    d = df[df["nopen"].notna()]
    if d.empty or not (d["session"] == "pm").any():
        return
    print("\n===== OVERNIGHT TEST (evening vs midday) =====")
    print("first trading day after the scan\n")
    for lab, ses in (("EVENING picks (~2:15 PM)", "pm"), ("MIDDAY picks (~11:30 AM)", "am")):
        g = d[d["session"] == ses]
        if g.empty:
            continue
        hi = g["nhigh"].dropna()
        print(lab, f"({len(g)})")
        print(f"   up at next open:      {(g['nopen'] > 0).mean():.0%}")
        print(f"   avg at next open:     {g['nopen'].mean():+.0%}")
        if len(hi):
            print(f"   avg best next day:    {hi.mean():+.0%}")
            print(f"   doubled by next day:  {(hi >= 1).mean():.0%}")
    print("   (option daily bars; the open is the first trade of the day)")


def print_lottery(df):
    """Lottery list: did the cheap contract DOUBLE at any point before expiry?
    Graded as 'sell the whole thing at +100% if it gets there, else ride to expiry'."""
    g = df[df["style"] == "lotto"]
    if g.empty:
        return
    done = g[g["expired"] | (g["peak_ret"] >= 1)]
    print("\n===== LOTTERY LIST (since 11/16) =====")
    print("cheap, 1-7 days out | sell at +100%\n")
    for _, r in g.iterrows():
        hit = "DOUBLED" if r["peak_ret"] >= 1 else ("expired" if r["expired"] else "open")
        print(f"{r['ticker']} {r['side'].upper()} {r['strike']:g}  exp {r['expiry']:%m/%d}"
              f" | paid ${r['entry'] * 100:,.0f} | best {r['peak_ret']:+.0%} | {hit}")
    if len(done):
        doubled = done["peak_ret"] >= 1
        res = doubled * 1.0 + (~doubled) * done["final_ret"].clip(lower=-1)
        print(f"\n   doubled at some point: {doubled.mean():.0%} ({int(doubled.sum())} of {len(done)})")
        print("   (break-even is about 50% if the misses go to zero)")
        print(f"   if $50 each, sell at +100%: {money(float(res.sum()) * 50)}")
        if "with_sector" in done.columns:
            for lab in ("with", "against"):
                part = done[done["with_sector"].fillna("") == lab]
                if len(part):
                    print(f"   {lab} sector flow: doubled {(part['peak_ret'] >= 1).mean():.0%} ({len(part)})")


def money(x):
    return f"{'+' if x >= 0 else '-'}${abs(x):,.0f}"


def run_review(today):
    if not os.path.exists(LOG_FILE):
        sys.exit("No log yet. Run a scan first.")
    log = pd.read_csv(LOG_FILE, parse_dates=["scan_date", "expiry"])
    if "style" not in log.columns:
        log["style"] = "short"
    log["style"] = log["style"].fillna("short")
    log = log[log["scan_date"].dt.weekday < 5]   # ignore any weekend scans (stale data)
    # If you scanned more than once in a day, keep only that day's last pick per stock.
    if "session" not in log.columns:
        log["session"] = "am"
    log["session"] = log["session"].fillna("am")
    log = log.drop_duplicates(subset=["scan_date", "session", "style", "ticker"], keep="last")
    # 11/6 9:03am: an old copy of the scanner logged non-box picks by mistake. Skip those.
    ver = log["ver"] if "ver" in log.columns else pd.Series(pd.NA, index=log.index)
    bs = log["box_state"] if "box_state" in log.columns else pd.Series("", index=log.index)
    stray = (log["scan_date"].dt.date == dt.date(2026, 11, 6)) & ver.isna() \
        & ~bs.isin(["retest", "extended"])
    if stray.any():
        print(f"(skipped {int(stray.sum())} picks from the old copy, 11/6 9:03am)")
    log = log[~stray]
    out = []
    for sym, g in log.groupby("ticker"):
        start = g["scan_date"].min().date()
        tk = yf.Ticker(sym)
        chains = {}
        try:
            hist = tk.history(start=start, end=today + dt.timedelta(days=1), auto_adjust=False)
        except Exception:
            continue
        if hist.empty:
            continue
        hist.index = [pd.Timestamp(d).date() for d in hist.index]
        for _, r in g.iterrows():
            scan_d, exp_d = r["scan_date"].date(), r["expiry"].date()
            w = hist[(hist.index > scan_d) & (hist.index <= min(exp_d, today))]
            if w.empty:
                continue
            k, entry = r["strike"], r["mid"]
            if r["side"] == "call":
                final, peak = max(0, w["Close"].iloc[-1] - k), max(0, w["High"].max() - k)
            else:
                final, peak = max(0, k - w["Close"].iloc[-1]), max(0, k - w["Low"].min())
            src = "expiry"
            oh = option_hist(r.get("contract"), scan_d, min(exp_d, today))
            if oh is not None:
                peak = max(peak, float(oh["High"].max()))
            if exp_d >= today:          # still open: use the real option price, not intrinsic
                live = live_option_price(tk, chains, r, exp_d)
                if live is not None:
                    final, src = live, "live"
                    peak = max(peak, final)
                else:
                    src = "intrinsic"
            fr = (final / entry - 1) if entry > 0 else 0
            pr = (peak / entry - 1) if entry > 0 else 0
            grade, gdate, gest = exit_grade(oh, entry, fr, pr, exp_d < today)
            grade2, _, _ = exit_grade(oh, entry, fr, pr, exp_d < today, CUT_WIDE)
            split = split_exit(oh, entry, grade, gdate, fr)
            earn_pick = "earnings" in str(r.get("why", ""))
            # Overnight test: what the contract did on the first trading day after the scan.
            nopen = nhigh = None
            if oh is not None and len(oh) and entry > 0:
                o1, h1 = nz(oh["Open"].iloc[0]), nz(oh["High"].iloc[0])
                nopen = o1 / entry - 1 if o1 > 0 else None
                nhigh = h1 / entry - 1 if h1 > 0 else None
            bs = r.get("box_state", "")
            out.append(dict(style=r["style"], box=bs if isinstance(bs, str) else "",
                            scan=scan_d, ticker=sym, side=r["side"], strike=k,
                            expiry=exp_d, entry=entry, expired=exp_d < today, src=src,
                            final_ret=fr, peak_ret=pr, score=r["score"],
                            grade=grade, gdate=gdate, gest=gest, grade2=grade2, split=split,
                            earn=earn_pick,
                            group=r.get("group", ""), with_sector=r.get("with_sector", ""),
                            session=r.get("session", "am"), nopen=nopen, nhigh=nhigh))
    if not out:
        print("Nothing to grade yet. Picks need at least")
        print("one trading day after the scan.")
        return
    df = pd.DataFrame(out).sort_values(["style", "scan", "score"], ascending=[False, False, False])
    global LAST_REVIEW
    LAST_REVIEW = df
    print("ZEN REVIEW")
    print("open picks: live option price")
    print("expired: value at expiration")
    print_top_runners(df)
    print_sector_test(df)
    print_lottery(df)
    print_overnight(df)
    for style in ("swing", "short"):
        g = df[df["style"] == style]
        if g.empty:
            continue
        print(f"\n===== {style.upper()} =====\n")
        for _, r in g.iterrows():
            status = "EXPIRED" if r["expired"] else "open"
            print(f"{r['ticker']} {r['side'].upper()} {r['strike']:g}  exp {r['expiry']:%m/%d}")
            print(f"   scanned {r['scan']:%m/%d} | score {r['score']:.0f}")
            print(f"   paid ${r['entry'] * 100:,.0f} | {status}")
            note = " (est.)" if r["src"] == "intrinsic" else ""
            print(f"   now {r['final_ret']:+.0%}{note} | best {r['peak_ret']:+.0%}")
            if r["grade"] == "WIN":
                when = f" on {r['gdate']:%m/%d}" if r["gdate"] else ""
                print(f"   RULES: WIN, hit +100%{when}" + (" (est.)" if r["gest"] else ""))
            elif r["grade"] == "LOSS":
                when = f" on {r['gdate']:%m/%d}" if r["gdate"] else ""
                print(f"   RULES: LOSS, hit -50%{when}" + (" (est.)" if r["gest"] else ""))
            elif r["grade"] == "EXPIRED":
                print(f"   RULES: expired between lines ({r['final_ret']:+.0%})")
            else:
                print("   RULES: still open")
            print()
        dec = g[g["grade"].isin(["WIN", "LOSS", "EXPIRED"])]
        if len(dec):
            w, l, n, dollars = rule_stats(dec, "grade", CUT_AT)
            still = int((g["grade"] == "open").sum())
            print(f"{style.upper()} EXIT-RULES SCORECARD")
            print("   (sell at +100% / cut at -50%)")
            print(f"   wins {w} | losses {l} | open {still}")
            print(f"   win rate: {w / n:.0%}  (break-even ~33%)")
            print(f"   if $100 per trade: {money(dollars)} on {n} trade{'s' if n != 1 else ''}")
            hi = dec[dec["score"] >= dec["score"].median()]
            lo = dec[dec["score"] < dec["score"].median()]
            if len(hi) and len(lo):
                print(f"   high scores win  {(hi['grade'] == 'WIN').mean():.0%}")
                print(f"   low scores win   {(lo['grade'] == 'WIN').mean():.0%}")
            bx = dec[dec["box"] == "retest"]
            if len(bx):
                print(f"   box retests win  {(bx['grade'] == 'WIN').mean():.0%} ({len(bx)})")
            # Earnings vs no earnings before expiry
            for label, part in (("earnings picks", dec[dec["earn"]]), ("no earnings", dec[~dec["earn"]])):
                if len(part):
                    pw, _, pn, pd_ = rule_stats(part, "grade", CUT_AT)
                    print(f"   {label:<15}win {pw / pn:.0%} ({pn}) {money(pd_)}")
            # Same picks with a looser -70% stop
            dec2 = g[g["grade2"].isin(["WIN", "LOSS", "EXPIRED"])]
            if len(dec2):
                w2, l2, n2, d2 = rule_stats(dec2, "grade2", CUT_WIDE)
                print("   with a -70% stop instead:")
                print(f"     wins {w2} | losses {l2} | win rate {w2 / n2:.0%}")
                print(f"     (break-even ~41%) | {money(d2)}")
            # Split exit: half sold at +100%, the rest rides until it falls back to entry
            sd = dec["split"].dropna()
            if len(sd):
                wins_s = dec[dec["grade"] == "WIN"]["split"].dropna()
                print("   split exit (half at +100%, rest rides,")
                print("   rest out if it falls back to entry):")
                print(f"     {money(float(sd.sum()) * 100)} on {len(sd)} trades")
                if len(wins_s):
                    print(f"     avg win {wins_s.mean():+.0%} (vs +100% all-out)")
            # How often the big runners show up (peak, before any exit)
            big5 = int((g["peak_ret"] >= 5).sum())
            big10 = int((g["peak_ret"] >= 10).sum())
            print(f"   peaked +500% or more:  {big5} of {len(g)}")
            print(f"   peaked +1000% or more: {big10} of {len(g)}")
            print()
        done = g[g["expired"]]
        if len(done):
            print(f"{style.upper()} HOLD-TO-EXPIRY ({len(done)} expired)")
            print(f"   profitable:      {(done['final_ret'] > 0).mean():.0%}")
            print(f"   doubled at peak: {(done['peak_ret'] >= 1).mean():.0%}")
            print(f"   went to zero:    {(done['final_ret'] <= -1).mean():.0%}")
            print(f"   avg return:      {done['final_ret'].mean():+.0%}")
            hi = done[done["score"] >= done["score"].median()]
            lo = done[done["score"] < done["score"].median()]
            if len(hi) and len(lo):
                print(f"   high scores win  {(hi['final_ret'] > 0).mean():.0%}")
                print(f"   low scores win   {(lo['final_ret'] > 0).mean():.0%}")
                print("   (if these match, the score")
                print("    has no edge yet)")
            bx = done[done["box"] == "retest"]
            other = done[done["box"] != "retest"]
            if len(bx):
                print(f"   box retests win  {(bx['final_ret'] > 0).mean():.0%} ({len(bx)})")
                if len(other):
                    print(f"   all others win   {(other['final_ret'] > 0).mean():.0%} ({len(other)})")


def main():
    ap = argparse.ArgumentParser(description="Zen Quant Scanner")
    ap.add_argument("--tickers", nargs="+", help="tickers to scan")
    ap.add_argument("--file", help="text file with one ticker per line")
    ap.add_argument("--style", choices=["both", "swing", "short"], default="both")
    ap.add_argument("--review", action="store_true", help="grade logged picks")
    a = ap.parse_args()
    today = market_today()
    if a.review:
        return run_review(today)
    tickers = a.tickers or DEFAULT_UNIVERSE
    if a.file:
        with open(a.file) as f:
            tickers = [t.strip().upper() for t in f if t.strip() and not t.startswith("#")]
    styles = ("swing", "short") if a.style == "both" else (a.style,)
    run_scan([t.upper() for t in tickers], today, styles)


if __name__ == "__main__":
    main()
