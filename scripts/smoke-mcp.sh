#!/usr/bin/env bash
set -euo pipefail

ENDPOINT="${ANVIL_MCP_ENDPOINT:-http://127.0.0.1:38765/mcp}"
TOKEN="${ANVIL_MCP_TOKEN:-}"
BASE_URL="${ENDPOINT%/mcp}"

AUTH_ARGS=()
if [[ -n "$TOKEN" ]]; then
  AUTH_ARGS=(-H "Authorization: Bearer $TOKEN")
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

rpc() {
  local body="$1"
  if [[ -n "$TOKEN" ]]; then
    curl -fsS "$ENDPOINT" \
      -H 'Content-Type: application/json' \
      -H 'Accept: application/json, text/event-stream' \
      -H 'MCP-Protocol-Version: 2025-06-18' \
      "${AUTH_ARGS[@]}" \
      -d "$body"
  else
    curl -fsS "$ENDPOINT" \
      -H 'Content-Type: application/json' \
      -H 'Accept: application/json, text/event-stream' \
      -H 'MCP-Protocol-Version: 2025-06-18' \
      -d "$body"
  fi
}

health() {
  if [[ -n "$TOKEN" ]]; then
    curl -fsS "$BASE_URL/health" "${AUTH_ARGS[@]}"
  else
    curl -fsS "$BASE_URL/health"
  fi
}

python_get_trace_ids='
import json
import sys

data = json.load(sys.stdin)
history = data.get("result", {}).get("structuredContent", {}).get("history", [])
ids = [row.get("id") for row in history if row.get("hasTrace") and row.get("id")]
print("\n".join(ids[:2]))
'

echo "MCP endpoint: $ENDPOINT"

echo "1. /health"
health >/dev/null

echo "2. GET /mcp stream probe"
STREAM_STATUS="$(curl -sS -o "$TMP_DIR/get-mcp.json" -w "%{http_code}" "$ENDPOINT" -H 'Accept: text/event-stream')"
if [[ "$STREAM_STATUS" != "405" ]]; then
  echo "Expected GET /mcp to return 405 when SSE is unavailable, got $STREAM_STATUS" >&2
  cat "$TMP_DIR/get-mcp.json" >&2
  exit 1
fi

echo "3. initialize"
rpc '{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "initialize",
  "params": {
    "protocolVersion": "2025-06-18",
    "capabilities": {},
    "clientInfo": { "name": "anvil-smoke-mcp", "version": "1.0" }
  }
}' >/dev/null

echo "4. tools/list"
rpc '{"jsonrpc":"2.0","id":2,"method":"tools/list"}' > "$TMP_DIR/tools.json"
grep -q '"list_history"' "$TMP_DIR/tools.json"
grep -q '"enhance_prompt"' "$TMP_DIR/tools.json"
grep -q '"write_lyrics"' "$TMP_DIR/tools.json"
grep -q '"mark_preferred_take"' "$TMP_DIR/tools.json"
grep -q '"apply_adapter"' "$TMP_DIR/tools.json"
grep -q '"generate_song"' "$TMP_DIR/tools.json"
grep -q '"generate"' "$TMP_DIR/tools.json"
grep -q '"generation_status"' "$TMP_DIR/tools.json"
grep -q '"release_memory"' "$TMP_DIR/tools.json"

echo "5. generate_song dry-run"
rpc '{
  "jsonrpc": "2.0",
  "id": 3,
  "method": "tools/call",
  "params": {
    "name": "generate_song",
    "arguments": {
      "prompt": "mcp smoke dry run",
      "duration": 5,
      "seed": 42,
      "pipeline": "aceStep",
      "dryRun": true
    }
  }
}' > "$TMP_DIR/generate-song-dry-run.json"
grep -q '"dryRun" : true' "$TMP_DIR/generate-song-dry-run.json"
grep -q '"submitted" : false' "$TMP_DIR/generate-song-dry-run.json"

echo "6. resources/list"
rpc '{"jsonrpc":"2.0","id":3,"method":"resources/list"}' > "$TMP_DIR/resources.json"
grep -Eq '"anvil:(\\/\\/|//)history"' "$TMP_DIR/resources.json"

echo "7. list_history"
rpc '{
  "jsonrpc": "2.0",
  "id": 4,
  "method": "tools/call",
  "params": {
    "name": "list_history",
    "arguments": { "limit": 10 }
  }
}' > "$TMP_DIR/history.json"

TRACE_IDS=()
while IFS= read -r trace_id; do
  if [[ -n "$trace_id" ]]; then
    TRACE_IDS+=("$trace_id")
  fi
done < <(/usr/bin/python3 -c "$python_get_trace_ids" < "$TMP_DIR/history.json")

if [[ "${#TRACE_IDS[@]}" -ge 1 ]]; then
  FIRST_ID="${TRACE_IDS[0]}"

  echo "8. get_trace ($FIRST_ID)"
  rpc "{
    \"jsonrpc\": \"2.0\",
    \"id\": 5,
    \"method\": \"tools/call\",
    \"params\": {
      \"name\": \"get_trace\",
      \"arguments\": { \"id\": \"$FIRST_ID\" }
    }
  }" >/dev/null

  echo "9. resources/read anvil://history/$FIRST_ID"
  rpc "{
    \"jsonrpc\": \"2.0\",
    \"id\": 6,
    \"method\": \"resources/read\",
    \"params\": { \"uri\": \"anvil://history/$FIRST_ID\" }
  }" > "$TMP_DIR/history-resource.json"
  grep -q '"contents"' "$TMP_DIR/history-resource.json"

  echo "10. resources/read anvil://trace/$FIRST_ID"
  rpc "{
    \"jsonrpc\": \"2.0\",
    \"id\": 7,
    \"method\": \"resources/read\",
    \"params\": { \"uri\": \"anvil://trace/$FIRST_ID\" }
  }" > "$TMP_DIR/trace-resource.json"
  grep -q '"contents"' "$TMP_DIR/trace-resource.json"
else
  echo "8-10. skipped trace checks; no traced history rows found"
fi

if [[ "${#TRACE_IDS[@]}" -ge 2 ]]; then
  FIRST_ID="${TRACE_IDS[0]}"
  SECOND_ID="${TRACE_IDS[1]}"

  echo "11. compare_traces ($FIRST_ID vs $SECOND_ID)"
  rpc "{
    \"jsonrpc\": \"2.0\",
    \"id\": 8,
    \"method\": \"tools/call\",
    \"params\": {
      \"name\": \"compare_traces\",
      \"arguments\": {
        \"left\": \"$FIRST_ID\",
        \"right\": \"$SECOND_ID\"
      }
    }
  }" >/dev/null
else
  echo "11. skipped compare; fewer than two traced history rows found"
fi

echo "12. legacy anvil.list_history alias"
rpc '{
  "jsonrpc": "2.0",
  "id": 9,
  "method": "tools/call",
  "params": {
    "name": "anvil.list_history",
    "arguments": { "limit": 1 }
  }
}' >/dev/null

echo "13. legacy anvil_list_history alias"
rpc '{
  "jsonrpc": "2.0",
  "id": 10,
  "method": "tools/call",
  "params": {
    "name": "anvil_list_history",
    "arguments": { "limit": 1 }
  }
}' >/dev/null

echo "MCP smoke test passed."
