"""Zen Scanner on GitHub: runs the stock scan on weekdays (~11:30 Central), the futures scan
on Asia-session evenings (Sun-Thu ~5:30 PM and ~9:00 PM Central) and the review on Saturdays,
then writes README.md / FUTURES.md / REVIEW.md and rebuilds the website (docs/)."""
import contextlib
import datetime as dt
import io
import os
import sys
from zoneinfo import ZoneInfo

import zen_scanner as z
import zen_futures as zf
import zen_site

z.LOG_FILE = "zen_log.csv"
z.SHOW_DETAILS = False
now = dt.datetime.now(ZoneInfo("America/Chicago"))
manual = os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch"
mode = os.environ.get("ZEN_MODE", "auto").strip().lower()
stamp = now.strftime("%a %m/%d %I:%M %p Central")

# Asia-session futures scans, Sunday-Thursday evenings (Central). Each schedule has a
# summer and a winter copy in zen.yml; only the copy that lands inside a window runs.
FUT_SLOTS = [("asia open", dt.time(17, 0), dt.time(18, 15)),
             ("asia mid", dt.time(20, 30), dt.time(21, 45))]
ASIA_DAYS = (6, 0, 1, 2, 3)   # Sun, Mon, Tue, Wed, Thu


def futures_slot():
    if now.weekday() not in ASIA_DAYS:
        return None
    for name, a, b in FUT_SLOTS:
        if a <= now.time() <= b:
            return name
    return None


if mode == "auto":
    slot = futures_slot()
    if now.weekday() == 5:
        mode = "review"
    elif slot:
        mode = "futures"
    elif now.weekday() <= 4:
        mode = "scan"
    else:
        print("Nothing scheduled now (Central):", stamp)
        sys.exit(0)


def capture(fn, *a):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            fn(*a)
        except SystemExit as e:
            print(e)
    return buf.getvalue()


def capture_safe(fn, *a):
    """Like capture, but a futures error is printed instead of failing the whole run."""
    try:
        return capture(fn, *a)
    except Exception as e:
        return f"{fn.__name__} failed: {e}"


if mode == "alert-test":
    zf.send_test_alert(now)
    sys.exit(0)

if mode == "futures":
    slot = futures_slot() or "manual"
    if not manual:
        key = f"{now:%Y-%m-%d} {slot}"
        if zen_site.load().get("futures_key") == key:
            print("Futures already scanned for", key)
            sys.exit(0)
    out = capture(zf.run_futures, now, slot)
    open("FUTURES.md", "w").write(
        f"# Zen Futures (Asia session)\n\n**{slot}** · {stamp}\n\n```\n{out.strip()}\n```\n\n"
        "Entry = broken box edge, stop = box 50%, target = next box. "
        "Research tool, not financial advice.\n")
    if zf.LAST_RUN:
        try:
            zen_site.update_futures(zf, now)
            if not manual:
                d = zen_site.load()
                d["futures_key"] = f"{now:%Y-%m-%d} {slot}"
                zen_site.save(d)
        except Exception as e:
            print("website futures update failed:", e)
elif mode == "scan":
    if not manual:
        # Two schedules cover daylight-saving time; only the one near 11:30 Central runs.
        if not (dt.time(11, 0) <= now.time() <= dt.time(14, 0)):
            print("Not scan time yet (Central):", stamp)
            sys.exit(0)
        if os.path.exists("README.md") and f"Scan {now:%Y-%m-%d}" in open("README.md").read():
            print("Already scanned today.")
            sys.exit(0)
    out = capture(z.run_scan, list(z.DEFAULT_UNIVERSE), z.market_today())
    body = (f"# Zen Scanner\n\n**Scan {now:%Y-%m-%d}** · {stamp}\n\n"
            f"```\n{out.strip()}\n```\n\n"
            "[Latest weekend review](REVIEW.md) · [Futures](FUTURES.md) · "
            "Research tool, not financial advice.\n")
    open("README.md", "w").write(body)
    if z.LAST_SCAN:
        try:
            zen_site.update_scan(z, now)
        except Exception as e:
            print("website scan update failed:", e)
else:
    out = capture(z.run_review, z.market_today())
    fout = capture_safe(zf.run_review, now)
    open("REVIEW.md", "w").write(f"# Zen Review\n\n{stamp}\n\n```\n{out.strip()}\n```\n\n"
                                 f"## Futures\n\n```\n{fout.strip()}\n```\n")
    try:
        zen_site.update_review(z, now)
    except Exception as e:
        print("website review update failed:", e)
    try:
        zen_site.update_futures_review(zf, now)
    except Exception as e:
        print("website futures review update failed:", e)
    out = out + "\n" + fout
try:
    zen_site.render()
except Exception as e:
    print("website render failed:", e)
print(out)
