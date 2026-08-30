"""Tiny HTTP server for interactive plotly charts, started in-thread by the bot.

  /                      index
  /sessions.html         full NDX session chart (hover: OHLC, ret, VIX, pattern, events)
  /day.html?date=...     1-min candlestick for a recent session (from nq_1m.csv)
  /static/<file>         static mounts: nq_analysis charts + this repo's charts/

Binds 0.0.0.0 so phone-on-same-WiFi can open links too (LAN only, no auth by design).
"""
from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CHARTS = ROOT / "charts"
NQ_ANALYSIS = Path("D:/nakuri_agent/nq_analysis")
PORT = 8791


def _sessions_fig(years: int = 3):
    import duckdb
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    con = duckdb.connect(str(ROOT / "data" / "nq_research.duckdb"), read_only=True)
    df = con.execute(
        "SELECT date, close, ret, gap, vix, vix_term_structure, three_candle_pattern, "
        "is_opex, is_triple_witching, days_to_fomc, consecutive_down_days "
        f"FROM sessions WHERE date >= (SELECT MAX(date) - INTERVAL {int(years)} YEAR FROM sessions) "
        "ORDER BY date"
    ).fetchdf()
    con.close()
    d = pd.to_datetime(df["date"])

    fig = make_subplots(rows=2, cols=1, row_heights=[0.72, 0.28], shared_xaxes=True, vertical_spacing=0.04)
    fig.add_trace(go.Candlestick(
        x=d, open=df["close"] * (1 - df["gap"].fillna(0)),
        high=df["close"] * 1.004, low=df["close"] * 0.996, close=df["close"], name="NDX",
        hovertemplate="%{x|%Y-%m-%d}<br>close %{y:,.0f}<extra></extra>"), row=1, col=1)
    # event markers
    ev = df[df["is_opex"]]
    fig.add_trace(go.Scatter(x=pd.to_datetime(ev["date"]), y=ev["close"], mode="markers",
                             marker=dict(symbol="diamond", size=7, color="purple"),
                             name="OPEX", hovertemplate="OPEX %{x|%Y-%m-%d}<extra></extra>"), row=1, col=1)
    ev = df[df["is_triple_witching"]]
    fig.add_trace(go.Scatter(x=pd.to_datetime(ev["date"]), y=ev["close"], mode="markers",
                             marker=dict(symbol="star", size=9, color="orange"),
                             name="witching", hovertemplate="triple witching %{x|%Y-%m-%d}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Bar(x=d, y=df["ret"] * 100, name="daily ret %",
                         marker_color=["green" if v >= 0 else "red" for v in df["ret"]],
                         hovertemplate="%{x|%Y-%m-%d}<br>%{y:+.2f}%<br>VIX %{customdata[0]:.1f}"
                                       "<br>pattern %{customdata[1]}<extra></extra>",
                         customdata=df[["vix", "three_candle_pattern"]]), row=2, col=1)
    fig.update_layout(height=760, title=f"NDX sessions — last {years}y (hover for detail; OPEX ◆, witching ★)",
                      xaxis_rangeslider_visible=False, margin=dict(t=60), showlegend=True)
    fig.update_yaxes(title_text="close", row=1, col=1)
    fig.update_yaxes(title_text="ret %", row=2, col=1)
    return fig


def _day_fig(day: str):
    import plotly.graph_objects as go

    csv = NQ_ANALYSIS_1M = Path(_env1m())
    if not csv.exists():
        raise FileNotFoundError(f"1m csv not configured: {csv}")
    m1 = pd.read_csv(csv, parse_dates=["Datetime"])
    m1["dt"] = pd.to_datetime(m1["Datetime"], utc=True).dt.tz_convert("America/New_York")
    sub = m1[m1["dt"].dt.date == pd.Timestamp(day).date()].reset_index(drop=True)
    if sub.empty:
        raise FileNotFoundError(f"no 1-min bars for {day} in the 7-day cache")
    fig = go.Figure(go.Candlestick(
        x=sub["dt"].dt.tz_localize(None), open=sub["Open"], high=sub["High"], low=sub["Low"], close=sub["Close"],
        increasing_line_color="green", decreasing_line_color="red",
        hovertemplate="%{x|%H:%M} ET<br>O %{open:,.1f} H %{high:,.1f}<br>L %{low:,.1f} C %{close:,.1f}"
                      "<extra></extra>"))
    # volume as secondary
    fig.add_trace(go.Bar(x=sub["dt"].dt.tz_localize(None), y=sub["Volume"], name="vol", yaxis="y2",
                         marker_color="rgba(100,120,200,0.5)",
                         hovertemplate="%{x|%H:%M} vol %{y:,}<extra></extra>"))
    fig.update_layout(height=720, title=f"NQ=F 1-min — {day} (ET)",
                      yaxis=dict(title="price"), yaxis2=dict(overlaying="y", side="right", title="vol"),
                      xaxis_rangeslider_visible=False, margin=dict(t=60))
    return fig


def _env1m() -> str:
    env = {}
    p = ROOT / ".env"
    if p.exists():
        for line in p.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env.get("NQ_1M_CSV", "")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _send_html(self, html: str, code: int = 200):
        data = html.encode()
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            if u.path in ("/", "/index.html"):
                links = "".join(f'<li><a href="/sessions.html?years={y}">NDX sessions {y}y</a></li>' for y in (1, 3, 10, 26))
                html = f"<h2>nq-research charts (interactive — hover/zoom)</h2><ul>{links}<li><a href='/day.html?date=2026-08-28'>Friday 1-min</a></li><li><a href='/static/nq_analogs.html'>historical analogs</a></li><li><a href='/static/nq_tech.html'>tech+GEX dashboard</a></li></ul>"
                return self._send_html(html)
            if u.path.startswith("/sessions.html"):
                years = int(q.get("years", ["3"])[0])
                return self._send_html(_sessions_fig(years).to_html(include_plotlyjs="cdn"))
            if u.path.startswith("/day.html"):
                day = q.get("date", ["2026-08-28"])[0]
                return self._send_html(_day_fig(day).to_html(include_plotlyjs="cdn"))
            if u.path.startswith("/static/"):
                name = Path(u.path).name
                for base in (NQ_ANALYSIS, CHARTS):
                    f = base / name
                    if f.exists() and f.suffix in (".html", ".png"):
                        data = f.read_bytes()
                        self.send_response(200)
                        self.send_header("Content-Type", "text/html" if f.suffix == ".html" else "image/png")
                        self.send_header("Content-Length", str(len(data)))
                        self.end_headers()
                        return self.wfile.write(data)
                return self._send_html("404", 404)
            return self._send_html("404", 404)
        except Exception as e:
            return self._send_html(f"<h3>error</h3><pre>{e}</pre>", 500)


def start_server(port: int = PORT) -> ThreadingHTTPServer:
    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def lan_ip() -> str:
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    finally:
        s.close()


if __name__ == "__main__":
    s = start_server()
    print(f"chart server on http://{lan_ip()}:{PORT}  (Ctrl+C to stop)")
    import time
    while True:
        time.sleep(60)