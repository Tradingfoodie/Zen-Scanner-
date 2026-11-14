"""Zen paper-trading bot (Alpaca PAPER account only: fake money, real market quotes).

What it does:
  - After the ~2:15 PM evening scan: buys the lottery picks (up to 5) plus the top 3 other
    picks, 1 contract each. Skips picks with a warning (!) and anything over MAX_COST.
  - Right after each buy fills: places a standing GTC sell at +100% (sell into the pop).
  - At every check: sells anything down 50% or more, and closes contracts on their
    expiration day (so nothing is auto-exercised).
  - Logs every action to bot_log.csv and writes BOT.md.

Keys come from GitHub secrets ALPACA_KEY_ID / ALPACA_SECRET_KEY. This file only ever talks
to the PAPER endpoint; it refuses to run against anything else.
"""
import datetime as dt
import json
import os
import urllib.error
import urllib.request
from zoneinfo import ZoneInfo

import pandas as pd

BASE = "https://paper-api.alpaca.markets"     # paper only, never the live endpoint
LOG = "bot_log.csv"
MAX_COST = 150.0          # $ per contract
MAX_NEW_PER_DAY = 8
MAX_OPEN = 25
TAKE_PROFIT = 2.0         # sell at 2x entry (+100%)
STOP = 0.5                # sell at 0.5x entry (-50%)
TOP_OTHER = 3             # besides the lottery list, the top N picks by score
CT = ZoneInfo("America/Chicago")


def keys():
    k, s = os.environ.get("ALPACA_KEY_ID", "").strip(), os.environ.get("ALPACA_SECRET_KEY", "").strip()
    return (k, s) if k and s else None


def api(method, path, body=None, params=None):
    assert BASE.startswith("https://paper-api."), "bot is paper-only"
    k, s = keys()
    url = BASE + path
    if params:
        url += "?" + "&".join(f"{a}={b}" for a, b in params.items())
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"APCA-API-KEY-ID": k, "APCA-API-SECRET-KEY": s,
                                          "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            txt = r.read().decode()
            return json.loads(txt) if txt else {}
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Alpaca {method} {path}: {e.code} {e.read().decode()[:200]}")


def log(action, symbol, qty=0, price=None, note=""):
    row = dict(time=dt.datetime.now(CT).isoformat(timespec="seconds"), action=action,
               symbol=symbol, qty=qty, price=price, note=note)
    df = pd.DataFrame([row])
    df.to_csv(LOG, mode="a", header=not os.path.exists(LOG), index=False)
    print(f"BOT {action} {symbol} x{qty} {'' if price is None else f'@{price}'} {note}")


def occ_expiry(sym):
    """OCC option symbol ROOT + YYMMDD + C/P + strike*1000 (8 digits) -> expiration date."""
    try:
        return dt.datetime.strptime(sym[-15:-9], "%y%m%d").date()
    except Exception:
        return None


def tick_round(p):
    """Options trade in $0.01 under $3 and $0.05 at $3+ (round up so the target is reached)."""
    if p < 3:
        return round(p + 1e-9, 2)
    return round(-(-p // 0.05) * 0.05, 2)


def option_positions():
    return [p for p in api("GET", "/v2/positions") if p.get("asset_class") == "us_option"]


def open_sells():
    orders = api("GET", "/v2/orders", params={"status": "open", "limit": 500})
    return {o["symbol"]: o for o in orders if o.get("side") == "sell"}


def sell_now(sym, qty, why, sells):
    if sym in sells:
        try:
            api("DELETE", f"/v2/orders/{sells[sym]['id']}")
        except Exception as e:
            print("cancel failed:", e)
    api("POST", "/v2/orders", dict(symbol=sym, qty=str(qty), side="sell", type="market",
                                    time_in_force="day"))
    log("SELL", sym, qty, None, why)


def manage(now):
    """Stops, expiration-day exits, and make sure every position has its +100% sell order."""
    sells = open_sells()
    for p in option_positions():
        sym, qty = p["symbol"], int(float(p["qty"]))
        if qty <= 0:
            continue                    # never act on a short / odd position
        if sym in sells and sells[sym].get("type") == "market":
            continue                    # a sell is already on its way; don't sell twice
        entry, cur = float(p["avg_entry_price"]), float(p.get("current_price") or 0)
        exp = occ_expiry(sym)
        if exp and exp <= now.date() and now.time() >= dt.time(13, 0):
            sell_now(sym, qty, "expiration day", sells)
            continue
        if cur > 0 and cur <= entry * STOP:
            sell_now(sym, qty, f"stop -50% (entry {entry:.2f}, now {cur:.2f})", sells)
            continue
        if sym not in sells:
            tp = tick_round(entry * TAKE_PROFIT)
            api("POST", "/v2/orders", dict(symbol=sym, qty=str(qty), side="sell", type="limit",
                                            limit_price=str(tp), time_in_force="gtc"))
            log("TAKE-PROFIT ORDER", sym, qty, tp, f"+100% of entry {entry:.2f}")


def choose(picks):
    """Lottery picks + top N others; skip warnings and expensive contracts."""
    p = picks.copy()
    p = p[p["contract"].fillna("").astype(str) != ""]
    p = p[~p["why"].fillna("").str.contains("WARNING")]
    p = p[p["mid"] * 100 <= MAX_COST]
    lot = p[p["style"] == "lotto"].sort_values("score", ascending=False)
    other = p[p["style"] != "lotto"].sort_values("score", ascending=False).head(TOP_OTHER)
    return pd.concat([lot, other]).drop_duplicates("contract")


def buy_picks(picks, now):
    held = {p["symbol"] for p in option_positions()}
    if len(held) >= MAX_OPEN:
        print("BOT: at the open-position limit, no new buys")
        return
    today = 0
    if os.path.exists(LOG):
        l = pd.read_csv(LOG)
        today = int(((l["action"] == "BUY") & l["time"].astype(str).str.startswith(now.date().isoformat())).sum())
    for _, r in choose(picks).iterrows():
        if today >= MAX_NEW_PER_DAY or len(held) >= MAX_OPEN:
            break
        sym = str(r["contract"])
        if sym in held:
            continue
        try:
            api("POST", "/v2/orders", dict(symbol=sym, qty="1", side="buy", type="market",
                                            time_in_force="day"))
            log("BUY", sym, 1, round(float(r["mid"]), 2),
                f"{r['style']} score {float(r['score']):.0f} (scanner mid)")
            held.add(sym)
            today += 1
        except Exception as e:
            log("BUY FAILED", sym, 1, None, str(e)[:120])


def report(now):
    acct = api("GET", "/v2/account")
    pos = option_positions()
    lines = [f"# Zen paper bot\n\n{now:%a %m/%d %I:%M %p} Central · Alpaca PAPER account (not real money)\n",
             f"Equity ${float(acct['equity']):,.2f} · cash ${float(acct['cash']):,.2f} "
             f"(started at ${float(acct.get('last_equity', acct['equity'])):,.2f} yesterday)\n",
             "## Open positions\n", "| Contract | Qty | Entry | Now | P&L |", "|---|---|---|---|---|"]
    for p in pos:
        lines.append(f"| {p['symbol']} | {p['qty']} | {float(p['avg_entry_price']):.2f} | "
                     f"{float(p.get('current_price') or 0):.2f} | {float(p['unrealized_pl']):+,.0f} "
                     f"({float(p['unrealized_plpc']):+.0%}) |")
    if not pos:
        lines.append("| none | | | | |")
    if os.path.exists(LOG):
        l = pd.read_csv(LOG).tail(15).iloc[::-1]
        lines += ["\n## Recent actions\n", "| Time | Action | Contract | Qty | Price | Note |",
                  "|---|---|---|---|---|---|"]
        lines += [f"| {r['time'][5:16].replace('T', ' ')} | {r['action']} | {r['symbol']} | {r['qty']} | "
                  f"{'' if pd.isna(r['price']) else r['price']} | {r['note'] if isinstance(r['note'], str) else ''} |"
                  for _, r in l.iterrows()]
    open("BOT.md", "w").write("\n".join(lines) + "\n")


def run(now, picks=None, session=None):
    """Called by run.py. picks = the evening scan's picks (buys only after the pm scan)."""
    if not keys():
        print("BOT: Alpaca paper keys not set (ALPACA_KEY_ID / ALPACA_SECRET_KEY), skipping.")
        return
    try:
        if not api("GET", "/v2/clock").get("is_open"):
            print("BOT: market closed, report only.")
        else:
            manage(now)
            if session == "pm" and picks is not None and not picks.empty:
                buy_picks(picks, now)
                import time
                time.sleep(15)          # let the market orders fill, then set the +100% sells
                manage(now)
        report(now)
    except Exception as e:
        print("BOT error:", e)
