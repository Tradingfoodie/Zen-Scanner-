"""Zen Scanner on GitHub: runs the scan on weekdays (~11:30 Central) and the
review on Saturdays, then writes README.md / REVIEW.md and rebuilds the website (docs/)."""
import contextlib
import datetime as dt
import io
import os
import sys
from zoneinfo import ZoneInfo

import zen_scanner as z
import zen_site

z.LOG_FILE = "zen_log.csv"
z.SHOW_DETAILS = False
now = dt.datetime.now(ZoneInfo("America/Chicago"))
manual = os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch"
mode = os.environ.get("ZEN_MODE", "auto")
if mode == "auto":
    mode = "review" if now.weekday() >= 5 else "scan"
stamp = now.strftime("%a %m/%d %I:%M %p Central")


def capture(fn, *a):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            fn(*a)
        except SystemExit as e:
            print(e)
    return buf.getvalue()


if mode == "scan":
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
            "[Latest weekend review](REVIEW.md) · Research tool, not financial advice.\n")
    open("README.md", "w").write(body)
    if z.LAST_SCAN:
        try:
            zen_site.update_scan(z, now)
        except Exception as e:
            print("website scan update failed:", e)
else:
    out = capture(z.run_review, z.market_today())
    open("REVIEW.md", "w").write(f"# Zen Review\n\n{stamp}\n\n```\n{out.strip()}\n```\n")
    try:
        zen_site.update_review(z, now)
    except Exception as e:
        print("website review update failed:", e)
try:
    zen_site.render()
except Exception as e:
    print("website render failed:", e)
print(out)
