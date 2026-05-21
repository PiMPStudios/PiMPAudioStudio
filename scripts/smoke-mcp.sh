#!/usr/bin/env bash
set -euo pipefail

ENDPOINT="${ANVIL_MCP_ENDPOINT:-http://127.0.0.1:38765/mcp}"
TOKEN="${ANVIL_MCP_TOKEN:-}"
BASE_URL="${ENDPOINT%/mcp}"

TMP_DIR="$(mktemp -d)"
SESSION_ID=""

cleanup_session() {
  if [[ -n "$SESSION_ID" ]]; then
    local curl_args=(
      -fsS
      -X DELETE
      "$ENDPOINT"
      -H 'Accept: application/json, text/event-stream'
      -H 'MCP-Protocol-Version: 2025-06-18'
      -H "MCP-Session-Id: $SESSION_ID"
    )
    if [[ -n "$TOKEN" ]]; then
      curl_args+=(-H "Authorization: Bearer $TOKEN")
    fi
    curl "${curl_args[@]}" >/dev/null 2>&1 || true
  fi
}

trap 'cleanup_session; rm -rf "$TMP_DIR"' EXIT

python_extract_mcp_json='
import json
import sys

text = sys.stdin.read()
if not text.strip():
    sys.exit("MCP response did not include a JSON-RPC body")

try:
    json.loads(text)
    print(text)
    sys.exit(0)
except json.JSONDecodeError:
    pass

events = []
buffer = []
for line in text.splitlines():
    if line.startswith("data:"):
        buffer.append(line[5:].lstrip())
    elif line == "" and buffer:
        payload = "\n".join(buffer).strip()
        buffer = []
        if not payload:
            continue
        try:
            events.append(json.loads(payload))
        except json.JSONDecodeError:
            pass

if buffer:
    payload = "\n".join(buffer).strip()
    if payload:
        try:
            events.append(json.loads(payload))
        except json.JSONDecodeError:
            pass

if not events:
    sys.exit("MCP SSE response did not include a JSON-RPC data event")

print(json.dumps(events[-1], indent=2, sort_keys=True))
'

session_header() {
  /usr/bin/python3 - "$1" <<'PY'
import sys

headers = open(sys.argv[1], encoding="utf-8", errors="ignore").read().splitlines()
for line in headers:
    if ":" not in line:
        continue
    key, value = line.split(":", 1)
    if key.lower() == "mcp-session-id":
        print(value.strip())
        break
PY
}

rpc() {
  local body="$1"
  local request_id="$2"
  local headers_file="$TMP_DIR/rpc-$request_id.headers"
  local raw_file="$TMP_DIR/rpc-$request_id.raw"
  local curl_args=(
    -fsS
    "$ENDPOINT"
    -D "$headers_file"
    -H 'Content-Type: application/json'
    -H 'Accept: application/json, text/event-stream'
    -H 'MCP-Protocol-Version: 2025-06-18'
  )

  if [[ -n "$TOKEN" ]]; then
    curl_args+=(-H "Authorization: Bearer $TOKEN")
  fi
  if [[ -n "$SESSION_ID" ]]; then
    curl_args+=(-H "MCP-Session-Id: $SESSION_ID")
  fi

  curl "${curl_args[@]}" -d "$body" > "$raw_file"

  local new_session
  new_session="$(session_header "$headers_file")"
  if [[ -n "$new_session" ]]; then
    SESSION_ID="$new_session"
  fi

  /usr/bin/python3 -c "$python_extract_mcp_json" < "$raw_file"
}

health() {
  if [[ -n "$TOKEN" ]]; then
    curl -fsS "$BASE_URL/health" -H "Authorization: Bearer $TOKEN"
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

python_get_audio_ref='
import json
import os
import sys

data = json.load(sys.stdin)
history = data.get("result", {}).get("structuredContent", {}).get("history", [])
for row in history:
    history_id = row.get("id")
    path = row.get("audioFilePath")
    if history_id and path and os.path.isfile(path):
        print(json.dumps({"id": history_id, "path": path}))
        break
'

python_check_generate_dry_run='
import json
import sys

data = json.load(sys.stdin)
structured = data.get("result", {}).get("structuredContent", {})
generation = structured.get("generation", {})
if structured.get("dryRun") is not True:
    sys.exit("generate_song dry-run response did not set structuredContent.dryRun")
if structured.get("submitted") is not False:
    sys.exit("generate_song dry-run response unexpectedly submitted")
if generation.get("dryRun") is not True:
    sys.exit("generate_song dry-run response did not set generation.dryRun")
if generation.get("submitted") is not False:
    sys.exit("generate_song dry-run generation unexpectedly submitted")
if generation.get("operationID") is not None:
    sys.exit("generate_song dry-run generation unexpectedly returned operationID")
'

python_check_editor_inpaint_dry_run='
import json
import sys

data = json.load(sys.stdin)
structured = data.get("result", {}).get("structuredContent", {})
request = structured.get("request", {})
if structured.get("dryRun") is not True:
    sys.exit("submit_editor_inpaint dry-run response did not set dryRun")
if structured.get("submitted") is not False:
    sys.exit("submit_editor_inpaint dry-run unexpectedly submitted")
if structured.get("requiresLoadedAudio") is not True:
    sys.exit("submit_editor_inpaint dry-run did not advertise loaded-audio requirement")
if structured.get("requiresConfirmation") is not True:
    sys.exit("submit_editor_inpaint dry-run did not advertise confirmation requirement")
if request.get("mode") != "inpaint":
    sys.exit("submit_editor_inpaint dry-run did not resolve inpaint mode")
if request.get("regionPrompt") != "replace the selected phrase with a cleaner synth swell":
    sys.exit("submit_editor_inpaint dry-run did not echo regionPrompt")
'

echo "MCP endpoint: $ENDPOINT"

echo "1. /health"
health >/dev/null

echo "2. initialize"
rpc '{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "initialize",
  "params": {
    "protocolVersion": "2025-06-18",
    "capabilities": {},
    "clientInfo": { "name": "anvil-smoke-mcp", "version": "1.0" }
  }
}' 1 >/dev/null

if [[ -z "$SESSION_ID" ]]; then
  echo "initialize did not return MCP-Session-Id" >&2
  exit 1
fi

echo "3. tools/list"
rpc '{"jsonrpc":"2.0","id":2,"method":"tools/list"}' 2 > "$TMP_DIR/tools.json"
grep -q '"list_history"' "$TMP_DIR/tools.json"
grep -q '"enhance_prompt"' "$TMP_DIR/tools.json"
grep -q '"write_lyrics"' "$TMP_DIR/tools.json"
grep -q '"mark_preferred_take"' "$TMP_DIR/tools.json"
grep -q '"apply_adapter"' "$TMP_DIR/tools.json"
grep -q '"generate_song"' "$TMP_DIR/tools.json"
grep -q '"generate"' "$TMP_DIR/tools.json"
grep -q '"generation_status"' "$TMP_DIR/tools.json"
grep -q '"editor_status"' "$TMP_DIR/tools.json"
grep -q '"load_editor_audio"' "$TMP_DIR/tools.json"
grep -q '"submit_editor_inpaint"' "$TMP_DIR/tools.json"
grep -q '"list_operations"' "$TMP_DIR/tools.json"
grep -q '"get_operation"' "$TMP_DIR/tools.json"
grep -q '"cancel_operation"' "$TMP_DIR/tools.json"
grep -q '"release_memory"' "$TMP_DIR/tools.json"

echo "4. generate_song dry-run"
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
}' 3 > "$TMP_DIR/generate-song-dry-run.json"
/usr/bin/python3 -c "$python_check_generate_dry_run" < "$TMP_DIR/generate-song-dry-run.json"

echo "5. editor_status"
rpc '{
  "jsonrpc": "2.0",
  "id": 4,
  "method": "tools/call",
  "params": {
    "name": "editor_status",
    "arguments": {}
  }
}' 4 > "$TMP_DIR/editor-status.json"
grep -q '"supportedActions"' "$TMP_DIR/editor-status.json"

echo "6. submit_editor_inpaint dry-run"
rpc '{
  "jsonrpc": "2.0",
  "id": 5,
  "method": "tools/call",
  "params": {
    "name": "submit_editor_inpaint",
    "arguments": {
      "mode": "inpaint",
      "regionPrompt": "replace the selected phrase with a cleaner synth swell",
      "selectionStart": 0.5,
      "selectionEnd": 1.5,
      "strength": 0.6,
      "guidanceScale": 7,
      "dryRun": true
    }
  }
}' 5 > "$TMP_DIR/editor-inpaint-dry-run.json"
/usr/bin/python3 -c "$python_check_editor_inpaint_dry_run" < "$TMP_DIR/editor-inpaint-dry-run.json"

echo "7. list_operations"
rpc '{
  "jsonrpc": "2.0",
  "id": 6,
  "method": "tools/call",
  "params": {
    "name": "list_operations",
    "arguments": { "limit": 5 }
  }
}' 6 > "$TMP_DIR/operations.json"
grep -q '"operations"' "$TMP_DIR/operations.json"

echo "8. resources/list"
rpc '{"jsonrpc":"2.0","id":7,"method":"resources/list"}' 7 > "$TMP_DIR/resources.json"
grep -Eq '"anvil:(\\/\\/|//)history"' "$TMP_DIR/resources.json"
grep -Eq '"anvil:(\\/\\/|//)editor"' "$TMP_DIR/resources.json"
grep -Eq '"anvil:(\\/\\/|//)operations"' "$TMP_DIR/resources.json"

echo "9. list_history"
rpc '{
  "jsonrpc": "2.0",
  "id": 8,
  "method": "tools/call",
  "params": {
    "name": "list_history",
    "arguments": { "limit": 10 }
  }
}' 8 > "$TMP_DIR/history.json"

TRACE_IDS=()
while IFS= read -r trace_id; do
  if [[ -n "$trace_id" ]]; then
    TRACE_IDS+=("$trace_id")
  fi
done < <(/usr/bin/python3 -c "$python_get_trace_ids" < "$TMP_DIR/history.json")

AUDIO_REF="$(/usr/bin/python3 -c "$python_get_audio_ref" < "$TMP_DIR/history.json")"
if [[ -n "$AUDIO_REF" ]]; then
  echo "10. load_editor_audio dry-run"
  /usr/bin/python3 - "$AUDIO_REF" > "$TMP_DIR/load-editor-audio-request.json" <<'PY'
import json
import sys

ref = json.loads(sys.argv[1])
print(json.dumps({
    "jsonrpc": "2.0",
    "id": 9,
    "method": "tools/call",
    "params": {
        "name": "load_editor_audio",
        "arguments": {
            "id": ref["id"],
            "dryRun": True
        }
    }
}))
PY
  rpc "$(cat "$TMP_DIR/load-editor-audio-request.json")" 9 > "$TMP_DIR/load-editor-audio-dry-run.json"
  grep -q '"dryRun"' "$TMP_DIR/load-editor-audio-dry-run.json"
else
  echo "10. skipped load_editor_audio dry-run; no existing history audio file found"
fi

if [[ "${#TRACE_IDS[@]}" -ge 1 ]]; then
  FIRST_ID="${TRACE_IDS[0]}"

  echo "11. get_trace ($FIRST_ID)"
  rpc "{
    \"jsonrpc\": \"2.0\",
    \"id\": 10,
    \"method\": \"tools/call\",
    \"params\": {
      \"name\": \"get_trace\",
      \"arguments\": { \"id\": \"$FIRST_ID\" }
    }
  }" 10 >/dev/null

  echo "12. resources/read anvil://history/$FIRST_ID"
  rpc "{
    \"jsonrpc\": \"2.0\",
    \"id\": 11,
    \"method\": \"resources/read\",
    \"params\": { \"uri\": \"anvil://history/$FIRST_ID\" }
  }" 11 > "$TMP_DIR/history-resource.json"
  grep -q '"contents"' "$TMP_DIR/history-resource.json"

  echo "13. resources/read anvil://trace/$FIRST_ID"
  rpc "{
    \"jsonrpc\": \"2.0\",
    \"id\": 12,
    \"method\": \"resources/read\",
    \"params\": { \"uri\": \"anvil://trace/$FIRST_ID\" }
  }" 12 > "$TMP_DIR/trace-resource.json"
  grep -q '"contents"' "$TMP_DIR/trace-resource.json"
else
  echo "11-13. skipped trace checks; no traced history rows found"
fi

if [[ "${#TRACE_IDS[@]}" -ge 2 ]]; then
  FIRST_ID="${TRACE_IDS[0]}"
  SECOND_ID="${TRACE_IDS[1]}"

  echo "14. compare_traces ($FIRST_ID vs $SECOND_ID)"
  rpc "{
    \"jsonrpc\": \"2.0\",
    \"id\": 13,
    \"method\": \"tools/call\",
    \"params\": {
      \"name\": \"compare_traces\",
      \"arguments\": {
        \"left\": \"$FIRST_ID\",
        \"right\": \"$SECOND_ID\"
      }
    }
  }" 13 >/dev/null
else
  echo "14. skipped compare; fewer than two traced history rows found"
fi

echo "15. legacy anvil.list_history alias"
rpc '{
  "jsonrpc": "2.0",
  "id": 14,
  "method": "tools/call",
  "params": {
    "name": "anvil.list_history",
    "arguments": { "limit": 1 }
  }
}' 14 >/dev/null

echo "16. legacy anvil_list_history alias"
rpc '{
  "jsonrpc": "2.0",
  "id": 15,
  "method": "tools/call",
  "params": {
    "name": "anvil_list_history",
    "arguments": { "limit": 1 }
  }
}' 15 >/dev/null

echo "17. GET /mcp stream + progress notification"
STREAM_CURL_ARGS=(
  -sS
  --no-buffer
  --max-time 5
  -D "$TMP_DIR/get-mcp.headers"
  -o "$TMP_DIR/get-mcp.sse"
  -w "%{http_code}"
  "$ENDPOINT"
  -H 'Accept: text/event-stream'
  -H 'MCP-Protocol-Version: 2025-06-18'
  -H "MCP-Session-Id: $SESSION_ID"
)
if [[ -n "$TOKEN" ]]; then
  STREAM_CURL_ARGS+=(-H "Authorization: Bearer $TOKEN")
fi

set +e
curl "${STREAM_CURL_ARGS[@]}" > "$TMP_DIR/get-mcp.status" 2> "$TMP_DIR/get-mcp.stderr" &
STREAM_PID=$!
sleep 0.5
set -e

rpc '{
  "jsonrpc": "2.0",
  "id": 16,
  "method": "tools/call",
  "params": {
    "_meta": { "progressToken": "anvil-smoke-progress" },
    "name": "generation_status",
    "arguments": {}
  }
}' 16 >/dev/null

set +e
wait "$STREAM_PID"
STREAM_EXIT=$?
set -e

STREAM_STATUS="$(cat "$TMP_DIR/get-mcp.status")"
if [[ "$STREAM_STATUS" != "200" ]]; then
  echo "Expected GET /mcp to open an SSE stream, got HTTP $STREAM_STATUS" >&2
  cat "$TMP_DIR/get-mcp.stderr" >&2
  cat "$TMP_DIR/get-mcp.sse" >&2
  exit 1
fi
if [[ "$STREAM_EXIT" != "0" && "$STREAM_EXIT" != "28" ]]; then
  echo "GET /mcp stream probe failed with curl exit $STREAM_EXIT" >&2
  cat "$TMP_DIR/get-mcp.stderr" >&2
  cat "$TMP_DIR/get-mcp.sse" >&2
  exit 1
fi
grep -q 'notifications/progress' "$TMP_DIR/get-mcp.sse"
grep -q 'anvil-smoke-progress' "$TMP_DIR/get-mcp.sse"

echo "MCP smoke test passed."
