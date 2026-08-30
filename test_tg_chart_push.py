"""Smoke-test the two new bot flows end-to-end against the live chat."""
import sys, asyncio
sys.path.insert(0, ".")
from pathlib import Path
from nq_research.telegram_bot import TG, _env, CHART_HOST
from nq_research.chart_server import _sessions_fig

env = _env()
tg = TG(env["NQ_BOT_TOKEN"], env["NQ_CHAT_ID"])

# 1) render + send the sessions chart (same code path as /sessions.html?years=3 / /analog)
out = ROOT = Path(".")
out = Path("charts/_tg_sessions_3y.png")
out.parent.mkdir(parents=True, exist_ok=True)
tg.send("⏳ rendering NDX sessions 3y… (smoke test)")
_sessions_fig(3).write_image(out, width=1250, height=820, scale=1.3)
res = tg.send_photo(out, caption=f"NDX sessions 3y — static preview\nInteractive: http://{CHART_HOST}/sessions.html?years=3")
print("sendPhoto ok, message_id:", res["result"]["message_id"])

# 2) confirm sendPhoto photo sizes came back (proof of real upload)
sizes = [p["file_size"] for p in res["result"]["photo"]]
print("photo variants bytes:", sizes)
assert max(sizes) > 100_000, "photo suspiciously small"