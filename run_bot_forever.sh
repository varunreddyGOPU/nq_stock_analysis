#!/bin/bash
# nqstocks_bot keepalive: restart the poller if it dies
cd /d/ollama_agent/nq-research
while true; do
  uv run python -m nq_research.telegram_bot 2>&1
  echo "[keepalive] bot exited ($?), restarting in 10s"
  sleep 10
done
