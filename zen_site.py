"""Zen Quant Desk website: turns the latest scan/review into docs/index.html.

Data lives in docs/data.json so a scan and a review can each update their own part.
Everything on the page is real market data (Yahoo, ~15 min delayed). Research only.
"""
import datetime as dt
import html
import json
import math
import os
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

DOCS = "docs"
DATA = os.path.join(DOCS, "data.json")
DESK = [("NQ=F", "NQ", "Nasdaq-100 futures"), ("ES=F", "ES", "S&P 500 futures"),
        ("GC=F", "GC", "Gold futures"), ("^N225", "Nikkei", "Nikkei 225 (Tokyo)"),
        ("^HSI", "Hang Seng", "Hang Seng (Hong Kong)"), ("JPY=X", "USD/JPY", "Dollar / yen"),
        ("EURUSD=X", "EUR/USD", "Euro / dollar"), ("SPY", "SPY", "S&P 500 ETF"),
        ("QQQ", "QQQ", "Nasdaq-100 ETF")]
SECTOR_NAMES = {"XLK": "Tech", "XLC": "Communication", "XLY": "Discretionary",
                "XLP": "Staples", "XLV": "Health care", "XLF": "Financials",
                "XLE": "Energy", "XLI": "Industrials"}


def txt(x):
    return "" if x is None or (isinstance(x, float) and math.isnan(x)) else str(x)


def esc(x):
    return html.escape("" if x is None else str(x))


def num(x, default=None):
    try:
        x = float(x)
        return default if math.isnan(x) else x
    except (TypeError, ValueError):
        return default


def load():
    try:
        return json.load(open(DATA))
    except Exception:
        return {}


def save(d):
    os.makedirs(DOCS, exist_ok=True)
    json.dump(d, open(DATA, "w"), indent=1, default=str)


# ---- Desk: the April "Renaissance-style" signals, now on real hourly data ----
def desk_signals(yf):
    out = []
    try:
        data = yf.download([t for t, _, _ in DESK], period="30d", interval="60m",
                           group_by="ticker", auto_adjust=False, progress=False, threads=True)
    except Exception:
        return out
    for sym, name, label in DESK:
        try:
            df = data[sym].dropna()
            c, h, l = df["Close"], df["High"], df["Low"]
            if len(c) < 30:
                continue
            delta = c.diff()
            up = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
            dn = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
            rsi = float(100 - 100 / (1 + up.iloc[-1] / max(dn.iloc[-1], 1e-12)))
            mid, sd = c.rolling(20).mean(), c.rolling(20).std()
            pctb = float((c.iloc[-1] - (mid.iloc[-1] - 2 * sd.iloc[-1])) / (4 * sd.iloc[-1]))
            tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
            atr_ratio = float(tr.iloc[-1] / tr.rolling(14).mean().iloc[-2])
            ema9, ema21 = c.ewm(span=9).mean().iloc[-1], c.ewm(span=21).mean().iloc[-1]
            day = c[c.index.date == c.index[-1].date()]
            prev = c[c.index.date < c.index[-1].date()]
            chg = float(c.iloc[-1] / prev.iloc[-1] - 1) if len(prev) else 0.0
            out.append(dict(name=name, label=label, price=float(c.iloc[-1]), chg=chg,
                            rsi=rsi, pctb=pctb, atr=atr_ratio,
                            trend="up" if ema9 > ema21 else "down",
                            spark=[float(v) for v in c.iloc[-40:]], fx=sym.endswith("=X"),
                            bars=len(day)))
        except Exception:
            continue
    return out


def update_scan(z, now):
    d = load()
    s = z.LAST_SCAN
    picks = []
    if s.get("picks") is not None:
        for _, r in s["picks"].sort_values("score", ascending=False).head(12).iterrows():
            picks.append(dict(
                ticker=r["ticker"], side=r["side"], strike=num(r["strike"]),
                expiry=str(r["expiry"])[:10], cost=num(r["mid"], 0) * 100,
                score=num(r["score"], 0), style=r["style"], tags=z.pick_tags(r),
                group=txt(r.get("group", "")), with_sector=txt(r.get("with_sector", "")),
                price=num(r.get("price"))))
    watches = []
    for w in s.get("watches") or []:
        watches.append(dict(ticker=w["ticker"], status=w["status"], top=w["top"],
                            bot=w["bot"], mid=w["mid"], price=w["price"]))
    d["scan"] = dict(time=now.isoformat(), picks=picks, watches=watches,
                     rs={k: v for k, v in (s.get("rs") or {}).items()})
    try:
        d["desk"] = desk_signals(z.yf)
        d["desk_time"] = now.isoformat()
    except Exception:
        pass
    save(d)


def update_review(z, now):
    d = load()
    df = z.LAST_REVIEW
    if df is None or df.empty:
        return
    cards = []
    for style in ("short", "swing"):
        g = df[df["style"] == style]
        dec = g[g["grade"].isin(["WIN", "LOSS", "EXPIRED"])]
        if not len(dec):
            continue
        w, l, n, dollars = z.rule_stats(dec, "grade", z.CUT_AT)
        cards.append(dict(style=style, wins=w, losses=l, n=n, rate=w / n, dollars=dollars,
                          open=int((g["grade"] == "open").sum())))
    top = []
    for _, r in df.sort_values("peak_ret", ascending=False).head(6).iterrows():
        if r["peak_ret"] <= 0:
            continue
        top.append(dict(ticker=r["ticker"], side=r["side"], strike=num(r["strike"]),
                        expiry=str(r["expiry"])[:10], peak=num(r["peak_ret"], 0),
                        now=num(r["final_ret"], 0), style=r["style"]))
    sector = []
    if "group" in df.columns:
        t = df[df["grade"].isin(["WIN", "LOSS"]) & (df["group"].fillna("").astype(str) != "")]
        for (grp, side), g in t.groupby(["group", "side"]):
            sector.append(dict(group=grp, side=side, wins=int((g["grade"] == "WIN").sum()), n=len(g)))
    d["review"] = dict(time=now.isoformat(), cards=cards, top=top, sector=sector)
    save(d)


def update_futures(zf, now):
    """Store the latest futures scan (and refresh the market desk) in data.json."""
    d = load()
    r = zf.LAST_RUN
    d["futures"] = dict(time=now.isoformat(), slot=r.get("slot", ""),
                        setups=r.get("setups", []), watches=r.get("watches", []),
                        levels=r.get("levels", {}), missing=r.get("missing", []))
    try:
        desk = desk_signals(zf.z.yf)
        if desk:
            d["desk"], d["desk_time"] = desk, now.isoformat()
    except Exception:
        pass
    save(d)


def update_futures_review(zf, now):
    d = load()
    log = zf.LAST_REVIEW
    if log is None or log.empty:
        return
    done = log[log["grade"].isin(["WIN", "LOSS", "TIME", "NO FILL"])].tail(12)
    recent = []
    for _, r in done.iloc[::-1].iterrows():
        recent.append(dict(contract=r["contract"], tf=r["tf"], dir=int(r["dir"]),
                           time=str(r["scan_time"]), entry=num(r["entry"]), grade=r["grade"],
                           result_r=num(r["result_r"], 0)))
    d["futures_review"] = dict(time=now.isoformat(), stats=zf.review_stats(log), recent=recent)
    save(d)


# ---- HTML ----------------------------------------------------------------------
def spark_svg(vals, color):
    if not vals or len(vals) < 2:
        return ""
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1
    pts = " ".join(f"{i * 100 / (len(vals) - 1):.1f},{28 - (v - lo) / rng * 26:.1f}"
                   for i, v in enumerate(vals))
    return (f'<svg class="spark" viewBox="0 0 100 30" preserveAspectRatio="none">'
            f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.6" '
            f'vector-effect="non-scaling-stroke"/></svg>')


def meter(label, value_text, frac, tone):
    frac = max(0.0, min(1.0, frac))
    return (f'<div class="meter"><span class="m-label">{label}</span><span class="m-val">{value_text}</span>'
            f'<span class="m-track"><span class="m-fill {tone}" style="width:{frac * 100:.0f}%"></span></span></div>')


def fmt_time(iso):
    try:
        t = dt.datetime.fromisoformat(iso)
        return t.strftime("%a %b %-d · %-I:%M %p CT")
    except Exception:
        return "—"


def desk_html(desk):
    if not desk:
        return '<p class="empty">Market data will appear after the next scan.</p>'
    cards = []
    for q in desk:
        up = q["chg"] >= 0
        tone = "pos" if up else "neg"
        price = f'{q["price"]:.4f}' if q["fx"] else f'{q["price"]:,.2f}'
        rsi_tone = "pos" if q["rsi"] >= 60 else "neg" if q["rsi"] <= 40 else "neu"
        b_tone = "warn" if q["pctb"] >= 0.9 or q["pctb"] <= 0.1 else "neu"
        a_tone = "warn" if q["atr"] >= 1.5 else "neu"
        cards.append(f'''
      <article class="card desk">
        <header><div><h3>{esc(q["name"])}</h3><p class="sub">{esc(q["label"])}</p></div>
          <div class="px"><b>{price}</b><span class="{tone}">{"▲" if up else "▼"} {abs(q["chg"]):.2%}</span></div></header>
        {spark_svg(q["spark"], "var(--pos)" if up else "var(--neg)")}
        {meter("Momentum", f'RSI {q["rsi"]:.0f}', q["rsi"] / 100, rsi_tone)}
        {meter("Stretch", f'%B {q["pctb"]:.2f}', q["pctb"], b_tone)}
        {meter("Range", f'{q["atr"]:.1f}× ATR', q["atr"] / 3, a_tone)}
        <p class="trend">Trend <b class="{"pos" if q["trend"] == "up" else "neg"}">{q["trend"].upper()}</b> · hourly EMA 9/21</p>
      </article>''')
    return '<div class="grid desk-grid">' + "".join(cards) + "</div>"


def sector_html(rs):
    if not rs:
        return '<p class="empty">Sector flow appears after the next scan.</p>'
    rows = sorted(rs.items(), key=lambda kv: kv[1], reverse=True)
    big = max(abs(v) for _, v in rows) or 1
    out = []
    for f, v in rows:
        w = abs(v) / big * 50
        side = "right" if v >= 0 else "left"
        out.append(f'<div class="flow-row"><span class="f-name">{esc(SECTOR_NAMES.get(f, f))}</span>'
                   f'<span class="f-track"><span class="f-bar {"pos" if v >= 0 else "neg"} {side}" '
                   f'style="width:{w:.1f}%"></span></span>'
                   f'<span class="f-val {"pos" if v >= 0 else "neg"}">{v:+.1%}</span></div>')
    return '<div class="card flow">' + "".join(out) + '<p class="note">5-day return vs SPY. Green = money flowing in.</p></div>'


def picks_html(picks):
    if not picks:
        return '<p class="empty">No picks passed the filters in the last scan.</p>'
    rows = []
    for p in picks:
        side = "C" if p["side"] == "call" else "P"
        flow = {"with": '<span class="chip pos">with flow</span>',
                "against": '<span class="chip neg">against</span>'}.get(p.get("with_sector") or "", "")
        tags = "".join(f'<span class="tag t-{t if t != "!" else "warn"}">{esc(t)}</span>' for t in (p["tags"] or ""))
        rows.append(f'''<tr><td><b>{esc(p["ticker"])}</b></td>
          <td class="{"pos" if side == "C" else "neg"}">{side} {p["strike"]:g}</td>
          <td>{esc(p["expiry"][5:].replace("-", "/"))}</td><td>${p["cost"]:,.0f}</td>
          <td><span class="score">{p["score"]:.0f}</span><span class="sty">{"short" if p["style"] == "short" else "swing"}</span></td>
          <td>{tags}{flow}</td></tr>''')
    return ('<div class="card table-wrap"><table><thead><tr><th>Ticker</th><th>Contract</th><th>Exp</th>'
            '<th>Cost</th><th>Score</th><th>Tags</th></tr></thead><tbody>' + "".join(rows) + "</tbody></table>"
            '<p class="note">B box · E earnings · F strong flow · ! warning. Scores have shown no edge yet (11/6 verdict).</p></div>')


def boxes_html(watches):
    if not watches:
        return '<p class="empty">No boxes to watch in the last scan.</p>'
    breaking = [w for w in watches if w["status"] in ("up", "down")]
    inside = [w for w in watches if w["status"] == "inside"][:6]
    out = []
    for w in breaking:
        up = w["status"] == "up"
        edge = w["top"] if up else w["bot"]
        out.append(f'''<div class="box-card {"pos" if up else "neg"}"><div class="b-head"><b>{esc(w["ticker"])}</b>
          <span class="chip {"pos" if up else "neg"}">{"Breaking up" if up else "Breaking down"}</span></div>
          <p>Edge <b>{edge:,.2f}</b> · now {w["price"]:,.2f}</p><p class="note">Wait for the punchback to the edge.</p></div>''')
    for w in inside:
        rng = (w["top"] - w["bot"]) or 1
        pos = max(0, min(1, (w["price"] - w["bot"]) / rng))
        out.append(f'''<div class="box-card"><div class="b-head"><b>{esc(w["ticker"])}</b><span class="chip neu">In a box</span></div>
          <div class="range"><span class="r-dot" style="left:{pos * 100:.0f}%"></span><span class="r-mid"></span></div>
          <p class="r-lab"><span>{w["bot"]:,.2f}</span><span>{w["top"]:,.2f}</span></p></div>''')
    return '<div class="grid box-grid">' + "".join(out) + "</div>"


def review_html(rv):
    if not rv:
        return '<p class="empty">The scorecard updates after the Saturday review.</p>'
    cards = []
    for c in rv.get("cards", []):
        good = c["rate"] >= 0.33
        cards.append(f'''<div class="card stat"><p class="sub">{c["style"].upper()} list</p>
          <b class="big {"pos" if good else "neg"}">{c["rate"]:.0%}</b><p>win rate · {c["wins"]}W {c["losses"]}L · {c["open"]} open</p>
          <p class="{"pos" if c["dollars"] >= 0 else "neg"}">{"+" if c["dollars"] >= 0 else "−"}${abs(c["dollars"]):,.0f} at $100/trade</p>
          <p class="note">Break-even is about 33%.</p></div>''')
    top = "".join(f'<li><b>{esc(t["ticker"])}</b> {"C" if t["side"] == "call" else "P"} {t["strike"]:g} '
                  f'<span class="pos">best {t["peak"]:+.0%}</span> <span class="note">now {t["now"]:+.0%}</span></li>'
                  for t in rv.get("top", []))
    sec = "".join(f'<li>{esc(s["group"]).capitalize()} {esc(s["side"])}s: <b>{s["wins"]} of {s["n"]}</b> '
                  f'({s["wins"] / s["n"]:.0%})</li>' for s in rv.get("sector", []) if s["n"])
    sec = sec or '<li class="note">Sector test results start after the first tagged picks are graded (from 11/9).</li>'
    return (f'<div class="grid stat-grid">{"".join(cards)}</div>'
            f'<div class="grid two"><div class="card"><h3>Top runners</h3><ul class="list">{top}</ul></div>'
            f'<div class="card"><h3>Sector test (11/9–11/20)</h3><ul class="list">{sec}</ul>'
            f'<p class="note">To beat: defensive calls 67%, tech calls 18%.</p></div></div>')


def candles_svg(ch, lines=(), box=None, w=320, h=150):
    """Small candlestick chart. ch = {"bars": [[o,h,l,c],...], "span": (start, end)} with span
    counted in bars from the right edge; lines = [(price, label, css color)]; box = (top, bot)."""
    bars = (ch or {}).get("bars") or []
    if len(bars) < 5:
        return ""
    n = len(bars)
    vals = [v for b in bars for v in (b[1], b[2])] + [p for p, _, _ in lines]
    if box:
        vals += list(box)
    lo, hi = min(vals), max(vals)
    pad = (hi - lo) * 0.06 or 1
    lo, hi = lo - pad, hi + pad
    right = 74                                    # room for line labels
    cw = (w - right) / n
    y = lambda v: h - 4 - (v - lo) / (hi - lo) * (h - 8)
    out = [f'<svg class="chart" viewBox="0 0 {w} {h}" preserveAspectRatio="none" role="img">']
    if box:
        a, b = ch.get("span") or (n, 0)
        x0 = max(0, n - a) * cw
        x1 = max(0, n - b) * cw
        out.append(f'<rect x="{x0:.1f}" y="{y(box[0]):.1f}" width="{max(x1 - x0, 2):.1f}" '
                   f'height="{max(y(box[1]) - y(box[0]), 1):.1f}" fill="rgba(34,211,238,.10)" '
                   f'stroke="rgba(34,211,238,.45)" stroke-dasharray="3 3"/>')
    for i, (o, hh, ll, c) in enumerate(bars):
        x = i * cw + cw / 2
        col = "var(--pos)" if c >= o else "var(--neg)"
        top, bot = y(max(o, c)), y(min(o, c))
        out.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{y(hh):.1f}" y2="{y(ll):.1f}" stroke="{col}" stroke-width="1"/>'
                   f'<rect x="{x - cw * 0.32:.1f}" y="{top:.1f}" width="{cw * 0.64:.1f}" height="{max(bot - top, 0.8):.1f}" fill="{col}"/>')
    for p, label, col in lines:
        yy = y(p)
        out.append(f'<line x1="0" x2="{w - right + 2}" y1="{yy:.1f}" y2="{yy:.1f}" stroke="{col}" stroke-width="1" stroke-dasharray="4 3"/>'
                   f'<text x="{w - right + 5}" y="{yy + 3.5:.1f}" fill="{col}" font-size="10" font-family="Share Tech Mono,monospace">{esc(label)}</text>')
    out.append("</svg>")
    return "".join(out)


def levels_html(levels):
    if not levels:
        return ""
    cards = []
    for name, lv in levels.items():
        dec = 1 if name == "MGC" else 2
        f = lambda x: f"{x:,.{dec}f}"
        px = lv["price"]
        rows = sorted(lv["levels"], key=lambda l: -l["value"])
        items, placed = [], False
        for l in rows:
            if not placed and l["value"] < px:
                items.append(f'<li class="now"><span>▶ Price now</span><b>{f(px)}</b></li>')
                placed = True
            tone = "pos" if "high" in l["name"].lower() else "neg" if "low" in l["name"].lower() else ""
            items.append(f'<li><span>{esc(l["name"])}</span><b class="{tone}">{f(l["value"])}</b>'
                         f'<em>{round(l["value"] - px, dec) + 0.0:+,.{dec}f}</em></li>')
        if not placed:
            items.append(f'<li class="now"><span>▶ Price now</span><b>{f(px)}</b></li>')
        near = []
        if lv.get("above"):
            near.append(f'{esc(lv["above"]["name"])} {f(lv["above"]["dist"])} pts above')
        if lv.get("below"):
            near.append(f'{esc(lv["below"]["name"])} {f(lv["below"]["dist"])} pts below')
        cards.append(f'''<div class="card lv"><div class="b-head"><b>{esc(name)}</b><span class="sty">nearest: {" · ".join(near) or "—"}</span></div>
          <ul class="lv-list">{"".join(items)}</ul></div>''')
    return ('<h3 class="subh">Key levels</h3><div class="grid fut-grid">' + "".join(cards) + '</div>'
            '<p class="note">Prior day = last full Globex day (5 PM–4 PM). NY = 8:30 AM–3:00 PM CT session. '
            'Overnight = since the 5:00 PM CT reopen (on weekends, Thursday night into Friday). Numbers on the right = distance from price.</p>')


def futures_html(fu):
    if not fu:
        return '<p class="empty">Futures setups appear after the first Asia-session scan (Sun–Thu, about 5:30 PM and 9:00 PM CT).</p>'
    cards = []
    for c in fu.get("setups", []):
        long_ = c["dir"] == 1
        tone = "pos" if long_ else "neg"
        retest = c["state"] == "retest"
        dec = 1 if c["contract"] == "MGC" else 2
        f = lambda x: f"{x:,.{dec}f}"
        low_r = '<span class="tag t-warn">R under 1</span>' if c["r"] < 1 else ""
        cards.append(f'''<div class="box-card {tone}"><div class="b-head"><b>{esc(c["contract"])} <span class="sty">{esc(c["tf"])}</span></b>
          <span class="chip {tone}">{"READY" if retest else "WATCH"} · {"LONG" if long_ else "SHORT"}</span></div>
          <table class="ft"><tr><td>Entry</td><td><b>{f(c["entry"])}</b></td></tr>
          <tr><td>Stop</td><td class="neg">{f(c["stop"])}</td></tr>
          <tr><td>Target</td><td class="pos">{f(c["target"])}</td></tr></table>
          {candles_svg(c.get("chart"), [(c["target"], "TARGET", "var(--pos)"), (c["entry"], "ENTRY", "var(--accent)"), (c["stop"], "STOP", "var(--neg)")], (c["top"], c["bot"]))}
          <p><b class="score">{c["r"]:.1f}R</b> <span class="sty">target: {esc(c["kind"])}</span> {low_r}</p>
          <p class="note">Risk {c["risk_ticks"]} ticks · ${c["risk_micro"]:,.0f} per {esc(c["contract"])} · ${c["risk_full"]:,.0f} per {esc(c["full"])}</p>
          <p class="note">{"Price is at the entry now." if retest else f"Price ran {f(abs(c['price'] - c['entry']))} past the edge: wait for a pullback to {f(c['entry'])}."} Now {f(c["price"])}.</p></div>''')
    if not cards:
        cards.append('<p class="empty">No box breakouts retesting in MES, MNQ or MGC right now.</p>')
    watch = []
    for w in fu.get("watches", []):
        if w["status"] == "skip":
            watch.append(f'<div class="box-card"><div class="b-head"><b>{esc(w["contract"])} <span class="sty">{esc(w["tf"])}</span></b>'
                         f'<span class="chip neu">skipped</span></div><p class="note">{esc(w["note"])}</p></div>')
            continue
        dec = 1 if w["contract"] == "MGC" else 2
        up, dn = w["status"] == "up", w["status"] == "down"
        tone = "pos" if up else "neg" if dn else ""
        label = "Breaking up" if up else "Breaking down" if dn else "In a box"
        chart = candles_svg(w.get("chart"), [(w["top"], f'{w["top"]:,.{dec}f}', "var(--pos)"),
                                             (w["bot"], f'{w["bot"]:,.{dec}f}', "var(--neg)")], (w["top"], w["bot"]))
        watch.append(f'''<div class="box-card {tone}"><div class="b-head"><b>{esc(w["contract"])} <span class="sty">{esc(w["tf"])}</span></b>
          <span class="chip {tone or "neu"}">{label}</span></div>{chart}
          <p class="note">Box {w["bot"]:,.{dec}f}–{w["top"]:,.{dec}f} · {w["bars"]} bars · now {w["price"]:,.{dec}f}</p></div>''')
    watch_html = ('<h3 class="subh">Box watch</h3><div class="grid fut-grid">' + ("".join(watch) or
                  '<p class="empty">No tight boxes in MES, MNQ or MGC right now.</p>') + '</div>')
    miss = fu.get("missing") or []
    miss_html = (f'<p class="warn">No data from Yahoo this run for {esc(", ".join(miss))}. '
                 'It will be retried at the next scan.</p>') if miss else ""
    return (f'{miss_html}{levels_html(fu.get("levels"))}<h3 class="subh">Setups</h3><div class="grid fut-grid">{"".join(cards)}</div>{watch_html}'
            f'<p class="note">Scan {esc(fmt_time(fu.get("time", "")))} ({esc(fu.get("slot", ""))}) · Entry = broken box edge, '
            f'stop = box 50%, target = next box. 15-min and hourly bars, ~10–15 min delayed.</p>')


def futures_review_html(fr):
    if not fr or not fr.get("stats"):
        return '<p class="empty">The futures scorecard fills in once logged setups have played out (graded at each Saturday review).</p>'
    cards = []
    for s in fr["stats"]:
        dec = s["wins"] + s["losses"] + s["timed"]
        tone = "pos" if s["total_r"] >= 0 else "neg"
        cards.append(f'''<div class="card stat"><p class="sub">{esc(s["name"])}</p>
          <b class="big {tone}">{s["total_r"]:+.1f}R</b>
          <p>{s["wins"]}W {s["losses"]}L {s["timed"]} timed out · {s["filled"]} of {s["logged"]} filled</p>
          <p class="{tone}">{"+" if s["dollars_micro"] >= 0 else "−"}${abs(s["dollars_micro"]):,.0f} at 1 micro each</p>
          <p class="note">avg {s["avg_r"]:+.2f}R per filled trade · {s["open"]} still open</p></div>''')
    rec = "".join(f'<li><b>{esc(r["contract"])}</b> {esc(r["tf"])} {"long" if r["dir"] == 1 else "short"} '
                  f'<span class="{"pos" if r["result_r"] > 0 else "neg" if r["result_r"] < 0 else "note"}">{esc(r["grade"])} {r["result_r"]:+.1f}R</span> '
                  f'<span class="note">{esc(fmt_time(r["time"]))}</span></li>' for r in fr.get("recent", []))
    return (f'<div class="grid stat-grid">{"".join(cards)}</div>'
            f'<div class="grid two"><div class="card"><h3>Recent setups</h3><ul class="list">{rec}</ul></div></div>'
            f'<p class="note">Futures review {esc(fmt_time(fr.get("time", "")))} · stop or target first; '
            f'open at 2:00 AM CT = closed there; a bar touching both counts as the stop.</p>')


CSS = """
:root{--bg:#070b10;--panel:#0d131b;--panel2:#111a24;--line:#1c2836;--text:#d3e0ec;--dim:#6f8396;
--accent:#22d3ee;--pos:#2ee59d;--neg:#ff4d6d;--warn:#ffcc33;--violet:#a78bfa}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font:15px/1.5 "DM Sans",system-ui,sans-serif;min-height:100vh}
body:before{content:"";position:fixed;inset:0;pointer-events:none;
background-image:linear-gradient(rgba(34,211,238,.035) 1px,transparent 1px),linear-gradient(90deg,rgba(34,211,238,.035) 1px,transparent 1px);background-size:44px 44px}
main{position:relative;max-width:1180px;margin:0 auto;padding:22px 16px 60px}
.top{display:flex;flex-wrap:wrap;gap:12px;justify-content:space-between;align-items:flex-end;padding-bottom:18px;border-bottom:1px solid var(--line);margin-bottom:22px}
h1{font:900 clamp(20px,5vw,28px)/1.1 Orbitron,monospace;letter-spacing:.14em;color:var(--accent);text-shadow:0 0 22px rgba(34,211,238,.35)}
.top p{font:12px "Share Tech Mono",monospace;color:var(--dim);letter-spacing:.08em;margin-top:6px}
.live{display:flex;align-items:center;gap:8px;font:12px "Share Tech Mono",monospace;color:var(--dim)}
.dot{width:8px;height:8px;border-radius:50%;background:var(--pos);box-shadow:0 0 10px var(--pos);animation:p 2s infinite}
@keyframes p{50%{opacity:.35}}
section{margin-top:30px}
h2{font:700 12px Orbitron,monospace;letter-spacing:.22em;color:var(--dim);margin-bottom:12px;display:flex;gap:10px;align-items:center}
h2:after{content:"";flex:1;height:1px;background:var(--line)}
h3{font:700 15px Orbitron,monospace;letter-spacing:.1em}
.grid{display:grid;gap:12px}
.desk-grid{grid-template-columns:repeat(auto-fit,minmax(210px,1fr))}
.box-grid{grid-template-columns:repeat(auto-fit,minmax(170px,1fr))}
.stat-grid{grid-template-columns:repeat(auto-fit,minmax(200px,1fr))}
.two{grid-template-columns:repeat(auto-fit,minmax(260px,1fr));margin-top:12px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px;position:relative;overflow:hidden}
.desk:before{content:"";position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,var(--accent),transparent)}
.desk header{display:flex;justify-content:space-between;gap:8px}
.sub{font:11px "Share Tech Mono",monospace;color:var(--dim);letter-spacing:.06em}
.px{text-align:right;font-family:"Share Tech Mono",monospace}.px b{display:block;font-size:17px}.px span{font-size:12px}
.spark{width:100%;height:40px;margin:10px 0 8px}
.meter{display:grid;grid-template-columns:1fr auto;gap:3px 8px;font:11px "Share Tech Mono",monospace;margin-top:8px}.meter .m-track{grid-column:1/-1}
.m-label{color:var(--dim)}.m-val{text-align:right}
.m-track{height:6px;background:var(--panel2);border-radius:3px;overflow:hidden}
.m-fill{display:block;height:100%;border-radius:3px;background:var(--dim)}
.m-fill.pos{background:var(--pos)}.m-fill.neg{background:var(--neg)}.m-fill.warn{background:var(--warn)}.m-fill.neu{background:#3b5168}
.trend{font:11px "Share Tech Mono",monospace;color:var(--dim);margin-top:10px}
.pos{color:var(--pos)}.neg{color:var(--neg)}.warn{color:var(--warn)}
.flow-row{display:grid;grid-template-columns:110px 1fr 60px;gap:10px;align-items:center;margin:7px 0;font-size:13px}
.f-track{position:relative;height:10px;background:var(--panel2);border-radius:5px}
.f-track:before{content:"";position:absolute;left:50%;top:-3px;bottom:-3px;width:1px;background:var(--line)}
.f-bar{position:absolute;top:0;bottom:0;border-radius:5px}.f-bar.right{left:50%}.f-bar.left{right:50%}
.f-bar.pos{background:linear-gradient(90deg,rgba(46,229,157,.4),var(--pos))}.f-bar.neg{background:linear-gradient(270deg,rgba(255,77,109,.4),var(--neg))}
.f-val{text-align:right;font-family:"Share Tech Mono",monospace;font-size:12px}
.table-wrap{padding:0;overflow-x:auto}
table{width:100%;border-collapse:collapse;font:13px "Share Tech Mono",monospace;min-width:560px}
th{text-align:left;font-weight:400;color:var(--dim);font-size:11px;letter-spacing:.08em;padding:12px 14px;border-bottom:1px solid var(--line)}
td{padding:11px 14px;border-bottom:1px solid rgba(28,40,54,.6);white-space:nowrap}
tr:hover td{background:var(--panel2)}
.score{display:inline-block;min-width:30px;font-weight:700;color:var(--accent)}.sty{color:var(--dim);font-size:11px;margin-left:4px}
.tag,.chip{display:inline-block;font:11px "Share Tech Mono",monospace;padding:1px 7px;border-radius:999px;border:1px solid var(--line);margin-right:4px}
.t-B{color:var(--accent);border-color:rgba(34,211,238,.4)}.t-E{color:var(--violet);border-color:rgba(167,139,250,.4)}
.t-F{color:var(--warn);border-color:rgba(255,204,51,.4)}.t-warn{color:var(--neg);border-color:rgba(255,77,109,.4)}
.chip.pos{border-color:rgba(46,229,157,.45)}.chip.neg{border-color:rgba(255,77,109,.45)}.chip.neu{color:var(--dim)}
.table-wrap .note{padding:10px 14px}
.box-card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px;font-size:13px}
.box-card.pos{border-color:rgba(46,229,157,.35)}.box-card.neg{border-color:rgba(255,77,109,.35)}
.b-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px}
.range{position:relative;height:8px;margin:14px 2px 6px;border-radius:4px;background:linear-gradient(90deg,rgba(255,77,109,.35),#24384b,rgba(46,229,157,.35))}
.r-mid{position:absolute;left:50%;top:-3px;bottom:-3px;width:1px;background:var(--dim)}
.r-dot{position:absolute;top:50%;width:12px;height:12px;margin:-6px 0 0 -6px;border-radius:50%;background:var(--accent);box-shadow:0 0 10px var(--accent)}
.r-lab{display:flex;justify-content:space-between;font:11px "Share Tech Mono",monospace;color:var(--dim)}
.stat .big{display:block;font:900 34px Orbitron,monospace;margin:6px 0}
.list{list-style:none;margin-top:10px;font-size:13px}.list li{padding:6px 0;border-bottom:1px solid rgba(28,40,54,.6)}
.note{font-size:12px;color:var(--dim);margin-top:6px}
.empty{color:var(--dim);font-size:13px;padding:14px;border:1px dashed var(--line);border-radius:10px}
footer{margin-top:40px;padding-top:16px;border-top:1px solid var(--line);font:11px "Share Tech Mono",monospace;color:var(--dim);letter-spacing:.06em;line-height:1.8}
.fut-grid{grid-template-columns:repeat(auto-fit,minmax(240px,1fr))}.subh{font-size:12px;color:var(--dim);margin:18px 0 10px}.chart{width:100%;height:150px;margin:6px 0;display:block}.lv-list{list-style:none;margin-top:6px;font:13px "Share Tech Mono",monospace}.lv-list li{display:grid;grid-template-columns:1fr auto 64px;gap:8px;padding:5px 0;border-bottom:1px solid rgba(28,40,54,.6)}.lv-list em{font-style:normal;color:var(--dim);text-align:right;font-size:11px}.lv-list li.now{color:var(--accent);grid-template-columns:1fr auto;border-bottom:1px solid var(--accent)}.ft{min-width:0;width:100%;margin:8px 0;font-size:14px}.ft td{padding:3px 0;border:0}.ft td:last-child{text-align:right}
@media(max-width:520px){.flow-row{grid-template-columns:92px 1fr 52px}}
"""


def render():
    d = load()
    scan, rv = d.get("scan") or {}, d.get("review")
    built = dt.datetime.now(ZoneInfo("America/Chicago"))
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><meta name="robots" content="noindex">
<title>Zen Quant Desk</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@700;900&family=Share+Tech+Mono&family=DM+Sans:wght@400;600&display=swap" rel="stylesheet">
<style>{CSS}</style></head><body><main>
<div class="top"><div><h1>ZEN QUANT DESK</h1>
<p>REAL DATA · BOX SCANNER · SECTOR FLOW · FUTURES DESK</p></div>
<div class="live"><span class="dot"></span>Last scan {esc(fmt_time(scan.get("time", "")))} · futures {esc(fmt_time((d.get("futures") or {}).get("time", "")))}</div></div>

<section><h2>MARKET DESK</h2>{desk_html(d.get("desk"))}
<p class="note">Hourly bars, ~15 min delayed. The April dashboard's three signals, now on real prices. Untested: context, not trade calls.</p></section>

<section><h2>FUTURES · ASIA SESSION</h2>{futures_html(d.get("futures"))}</section>

<section><h2>SECTOR FLOW</h2>{sector_html(scan.get("rs"))}</section>

<section><h2>TODAY'S PICKS</h2>{picks_html(scan.get("picks"))}</section>

<section><h2>BOX WATCH</h2>{boxes_html(scan.get("watches"))}</section>

<section><h2>SCORECARD</h2>{review_html(rv)}
<p class="note">Latest review: {esc(fmt_time((rv or {}).get("time", "")))} · rules: sell at +100%, cut at −50%.</p></section>

<section><h2>FUTURES SCORECARD</h2>{futures_review_html(d.get("futures_review"))}</section>

<footer>RESEARCH TOOL · NOT FINANCIAL ADVICE · DATA: YAHOO FINANCE<br>
Page built {esc(built.strftime("%a %b %-d, %-I:%M %p CT"))}</footer>
</main></body></html>"""
    os.makedirs(DOCS, exist_ok=True)
    open(os.path.join(DOCS, "index.html"), "w").write(page)
    open(os.path.join(DOCS, ".nojekyll"), "w").write("")
