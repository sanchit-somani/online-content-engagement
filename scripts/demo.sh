#!/usr/bin/env bash
set -euo pipefail

echo "==> Installing & training"
make venv
make install
make train

echo "==> Starting API"
. .venv/bin/activate

uvicorn app.main:app --host 127.0.0.1 --port 8000 >/tmp/so_demo.log 2>&1 &
PID=$!

cleanup () {
  if kill -0 "$PID" >/dev/null 2>&1; then
    echo "==> Stopping server (pid=$PID)"
    kill "$PID" >/dev/null 2>&1 || true
  else
    echo "==> Server already stopped (pid=$PID)"
  fi
}
trap cleanup EXIT

echo "==> Waiting for /health..."
for i in {1..40}; do
  if curl -s http://127.0.0.1:8000/health >/dev/null 2>&1; then
    echo "==> Server is up"
    break
  fi
  sleep 0.25
done

# If server never came up, show logs and fail
if ! curl -s http://127.0.0.1:8000/health >/dev/null 2>&1; then
  echo "==> Server failed to start. Logs:"
  tail -n 80 /tmp/so_demo.log
  exit 1
fi

echo ""
echo "==> Normal request"
curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Why does my SQL query return duplicate rows?",
    "body": "Two tables, one join... <code>SELECT *</code>",
    "tags": ["sql"],
    "hour": 14,
    "weekday": 2
  }' | python -m json.tool

echo ""
echo "==> Junk request"
curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "title": "",
    "body": "",
    "tags": [],
    "hour": 12,
    "weekday": 2
  }' | python -m json.tool

echo ""
echo "==> Done. Logs at /tmp/so_demo.log"
