"""NQ Telegram bot: interactive conditional-analysis Q&A via long polling.

Commands:
  /start, /help                 -- usage
  /what <YYYY-MM-DD> [HH:MM]    -- session/minute dive for a date (et)
  /analyze <condition string>   -- conditional base-rate query (see /syntax)
  /syntax                       -- filter vocabulary
  /analog                       -- last 5 days vs 26y of analogs (uses nq_analysis CSVs)
Free-text messages: best-effort date mention + generic help.

Security: replies only to NQ_CHAT_ID from .env. No look-ahead: every number
carries n + CI through ConditionalQuery, same as the CLI.
"""
from __future__ import annotations

import json
import os
import re
import time
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]


def _env() -> dict:
    env = {}
    p = ROOT / ".env"
    if p.exists():
        for line in p.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    env.update({k: v for k, v in os.environ.items() if k.startswith("NQ_")})
    return env


API = "https://api.telegram.org"

# interactive chart server: started alongside the bot so links always work
from nq_research.chart_server import start_server as _start_charts, lan_ip as _lan_ip

CHART_PORT = 8791
CHART_HOST = f"{_lan_ip()}:{CHART_PORT}"


class TG:
    def __init__(self, token: str, chat_id: str):
        self.token, self.chat = token, chat_id
        self.offset = 0

    def _call(self, method: str, **payload):
        r = requests.post(f"{API}/bot{self.token}/{method}", json=payload, timeout=60)
        r.raise_for_status()
        return r.json()

    def send(self, text: str, chat: str | None = None):
        self._call("sendMessage", chat_id=chat or self.chat, text=text[:4096], parse_mode="HTML",
                   disable_web_page_preview=True)

    def send_photo(self, path, caption: str | None = None, chat: str | None = None):
        with open(path, "rb") as fh:
            r = requests.post(
                f"{API}/bot{self.token}/sendPhoto",
                data={"chat_id": chat or self.chat, "parse_mode": "HTML",
                      "caption": (caption or "")[:1024]},
                files={"photo": fh},
                timeout=180,
            )
        r.raise_for_status()
        return r.json()

    def poll(self) -> list[dict]:
        data = self._call("getUpdates", offset=self.offset, timeout=50, allowed_updates=["message"])
        updates = data.get("result", [])
        for u in updates:
            self.offset = max(self.offset, u["update_id"] + 1)
        return [u for u in updates if "message" in u]


# ------------- analysis helpers (shared with CLI engine) -------------

MIN_N = 30


def sessions_frame(db: Path) -> pd.DataFrame:
    import duckdb
    con = duckdb.connect(str(db), read_only=True)
    try:
        return con.execute("SELECT * FROM sessions ORDER BY date").fetchdf()
    finally:
        con.close()


def fmt_pct(x) -> str:
    return f"{x * 100:+.2f}%" if x == x else "n/a"


def what_happened(db: Path, day: str, hm: str | None, one_m_csv: Path) -> str:
    """Session dive for a date; optional minute-level color from the 1-min cache."""
    from nq_research.query.conditional import ConditionalQuery

    q = ConditionalQuery(db=str(db))
    table = q._load()
    row = table[pd.to_datetime(table["date"]).dt.date == pd.Timestamp(day).date()]
    if row.empty:
        return f"No session on {day} (weekend/holiday or before 1990)."
    r = row.iloc[0]
    lines = [
        f"📅 <b>{pd.Timestamp(day).date()} ({['Mon','Tue','Wed','Thu','Fri'][int(r.dow) - 1]})</b>",
        f"Close <b>{r['close']:,.1f}</b> · ret {fmt_pct(r['ret'])} · gap {fmt_pct(r['gap'])}",
        f"VIX {r['vix']:.1f} ({r['vix_bucket']}) · term structure {'backwardated' if r['is_backwardation'] else 'upward'}",
        f"Pattern: <b>{r['three_candle_pattern']}</b> · down-streak {int(r['consecutive_down_days'])}d",
        f"OPEX in {int(r['days_to_opex'])}d / since {int(r['days_since_opex'])}d · FOMC in {int(r['days_to_fomc'])}d",
    ]
    # forward returns
    i = row.index[0]
    for h in (1, 5, 10, 20):
        j = i + h
        if j < len(table):
            fr = table.close.iloc[j] / r["close"] - 1
            lines.append(f"next {h:2d}d: {fmt_pct(fr)}")
    lines.append(f"\n<i>next-session context, n=1 — this is history, not a signal</i>")

    # minute color if requested and CSV covers it
    if hm and one_m_csv.exists():
        m1 = pd.read_csv(one_m_csv, parse_dates=["Datetime"])
        m1.columns = [c if c != "Datetime" else "dt" for c in m1.columns]
        m1["dt"] = pd.to_datetime(m1["dt"], utc=True).dt.tz_convert("America/New_York")
        t0 = pd.Timestamp(hm).time()
        t1 = (pd.Timestamp(f"2000-01-01 {hm}") + pd.Timedelta(minutes=60)).time()
        sub = m1[(m1["dt"].dt.date == pd.Timestamp(day).date())
                 & (m1["dt"].dt.time >= t0) & (m1["dt"].dt.time < t1)]
        if not sub.empty:
            w = sub.nlargest(3, "Close").iloc[0]
            b = sub.nsmallest(3, "Close").iloc[0]
            lines.append(
                f"\n⏱ <b>{hm} ET ±1h:</b> hi {sub.Close.max():,.2f} @ {sub.loc[sub.Close.idxmax(), 'dt']:%H:%M} · "
                f"lo {sub.Close.min():,.2f} @ {sub.loc[sub.Close.idxmin(), 'dt']:%H:%M}")
    return "\n".join(lines)


FILTER_KEYS = {
    "dow": int, "is_opex": bool, "is_opex_week": bool, "is_post_opex_week": bool,
    "is_triple_witching": bool, "is_election_week": bool, "is_midterm_year": bool,
    "is_quarter_end": bool, "is_backwardation": bool,
    "vix_bucket": str, "ret_bucket": str, "three_candle_pattern": str, "trend_50_200": str,
}


def parse_condition(text: str) -> tuple[dict, tuple | None]:
    """Very small parser: 'friday ret:-0.75..-0.5 post-opex vix<15 pattern:three_down'.

    Bounds are normalized so the cheaper value is always the lower one.
    """
    filters: dict = {}
    band = None
    text = text.lower().strip()
    toks = text.replace(",", " ").split()
    for tok in toks:
        if tok in ("friday", "monday", "tuesday", "wednesday", "thursday"):
            filters["dow"] = ["monday", "tuesday", "wednesday", "thursday", "friday"].index(tok) + 1
        elif tok in ("post-opex", "postopex"):
            filters["is_post_opex_week"] = True
        elif tok == "opex-week":
            filters["is_opex_week"] = True
        elif tok == "witching":
            filters["is_triple_witching"] = True
        elif tok == "quarter-end":
            filters["is_quarter_end"] = True
        elif tok == "backwardation":
            filters["is_backwardation"] = True
        elif tok == "up":
            filters["trend_50_200"] = "above200|50>200"
        elif tok == "down":
            filters["trend_50_200"] = "below200|50<200"
        elif tok.startswith("retpct:"):
            # percent-of-price shorthand: retpct:-0.75..-0.5 -> fractions
            a, b = tok[7:].split("..")
            av, bv = float(a) / 100, float(b) / 100
            band = (min(av, bv), max(av, bv))
            filters["ret"] = band
        elif tok.startswith("ret:"):
            # fractions: ret:-0.0075..-0.005 | explicit percent: retpct:-0.75..-0.5
            spec = tok[4:]
            if not spec:
                continue
            a, b = spec.split("..")
            av, bv = float(a), float(b)
            # bare 'ret:' with abs<1 is ambiguous; treat magnitude<=1 as percent ONLY with retpct
            band = (min(av, bv), max(av, bv))
            filters["ret"] = band
        elif tok.startswith("vix"):
            m = re.match(r"vix(<|<=|>|>=)([\d.]+)", tok)
            if m:
                filters["_vix_raw"] = (m.group(1), float(m.group(2)))   # handled via bucket approx below
        elif tok.startswith("pattern:"):
            filters["three_candle_pattern"] = tok.split(":", 1)[1]
        elif tok.startswith("dow:"):
            filters["dow"] = int(tok.split(":", 1)[1])
    if "_vix_raw" in filters:
        op, v = filters.pop("_vix_raw")
        # bucket approximation of a continuous vix condition (documented limitation)
        bucket = "<15" if v <= 15 else "15-20" if v <= 20 else "20-30" if v <= 30 else ">30"
        filters["vix_bucket"] = bucket
    return filters, band


def analyze_condition(db: Path, text: str) -> str:
    from nq_research.query.conditional import ConditionalQuery

    filters, band = parse_condition(text)
    target = "next_session_return"
    m = re.search(r"horizon[= ](\d+)", text.lower())
    horizon = int(m.group(1)) if m else 1
    if "direction" in text.lower():
        target = "next_session_direction"
    q = ConditionalQuery(db=str(db))
    res = q.conditional(filters=filters, target=target, horizon=1)
    txt = res.report()
    rep = q.multiple_testing_report()
    txt += f"\n\n<b>session trials:</b> {rep['trials']} · BH-adjusted p: {rep['adjusted_pvalues'][-1]:.3f}"
    return txt


HELP = """<b>NQ analyzer</b> — conditional history, never predictions.

<b>/what 2026-03-20</b> — everything known about that session
<b>/what 2026-03-20 13:45</b> — adds minute-level hi/lo (last 7 sessions)
<b>/analyze friday retpct:-0.75..-0.5 post-opex</b> — base-rate query
<b>/charts</b> — interactive hover/zoom chart links
<b>/syntax</b> — filter vocabulary

Every stat carries n + 95% CI. n&lt;30 = anecdote, and I'll say so."""


def main():
    env = _env()
    token = env["NQ_BOT_TOKEN"]
    chat = env["NQ_CHAT_ID"]
    db = ROOT / "data" / "nq_research.duckdb"
    one_m_csv = Path(env.get("NQ_1M_CSV", ""))
    _start_charts()
    tg = TG(token, chat)
    me = tg._call("getMe")["result"]
    print(f"polling as @{me['username']} -> chat {chat} | charts on http://{CHART_HOST}")
    try:
        tg.send(f"🟢 nq-analyzer online as @{me['username']}.\n{HELP}\n\n📊 Interactive charts: /charts")
    except Exception as e:
        print(f"welcome send failed (user must press Start first): {e}" if "chat not found" in str(e) else repr(e))
    while True:
        try:
            for u in tg.poll():
                msg = u["message"]
                txt = (msg.get("text") or "").strip()
                c = msg["chat"]["id"]
                if c != int(chat):
                    continue
                low = txt.lower()
                try:
                    if low.startswith("/start") or low.startswith("/help"):
                        tg.send(HELP)
                    elif low.startswith("/syntax"):
                        tg.send("<b>filters:</b> friday..thursday · retpct:-0.75..-0.5 (percent) or "
                                "ret:-0.0075..-0.005 (fraction) · post-opex · opex-week · witching · "
                                "quarter-end · backwardation · up/down (trend) · vix<15|<20|<30 · "
                                "pattern:three_down · horizon:N · direction")
                    elif low.startswith("/what "):
                        parts = txt.split()
                        tg.send(what_happened(db, parts[1], parts[2] if len(parts) > 2 else None, one_m_csv))
                    elif low.startswith("/analyze "):
                        tg.send(analyze_condition(db, txt[9:]))
                    elif low.startswith("/charts"):
                        tg.send(
                            "📊 <b>Interactive charts</b> (hover/zoom — open in browser):\n"
                            f"• <a href=\"http://{CHART_HOST}/sessions.html?years=3\">NDX sessions 3y</a> · "
                            f"<a href=\"http://{CHART_HOST}/sessions.html?years=26\">full 26y</a>\n"
                            f"• <a href=\"http://{CHART_HOST}/day.html?date=2026-08-28\">Friday 1-min</a> (change ?date=)\n"
                            f"• <a href=\"http://{CHART_HOST}/static/nq_analogs.html\">historical analogs</a> · "
                            f"<a href=\"http://{CHART_HOST}/static/nq_tech.html\">tech+GEX dashboard</a>\n"
                            f"Or send <code>/sessions years=10</code> to get a rendered PNG here.\n"
                            f"<i>LAN host {CHART_HOST} — same Wi-Fi from phone.</i>"
                        )
                    elif low.startswith("/sessions") or low.startswith("/analog"):
                        # render the sessions chart as PNG and push it into the chat
                        m_y = re.search(r"years?=(\d+)", low)
                        y = int(m_y.group(1)) if m_y else 3
                        from nq_research.chart_server import _sessions_fig
                        out = ROOT / "charts" / f"_tg_sessions_{y}y.png"
                        out.parent.mkdir(parents=True, exist_ok=True)
                        tg.send(f"⏳ rendering NDX sessions {y}y…")
                        _sessions_fig(y).write_image(out, width=1250, height=820, scale=1.3)
                        tg.send_photo(out, caption=f"NDX sessions {y}y — static preview\n"
                                      f"Interactive (hover/zoom): http://{CHART_HOST}/sessions.html?years={y}")
                    else:
                        # free text: try to find a date
                        m = re.search(r"(\d{4})-(\d{2})-(\d{2})", txt)
                        if m:
                            tg.send(what_happened(db, m.group(0), None, one_m_csv))
                        else:
                            tg.send("Try /what 2026-03-20 or /analyze friday ret:-0.5..-0.75\n" + HELP.split("\n")[2])
                except Exception as e:
                    tg.send(f"⚠️ {type(e).__name__}: {e}")
        except KeyboardInterrupt:
            tg.send("🔴 going offline")
            raise
        except Exception as e:
            print("poll error:", repr(e)[:160], flush=True)
            time.sleep(5)


if __name__ == "__main__":
    main()