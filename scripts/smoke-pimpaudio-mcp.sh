#!/usr/bin/env bash
set -euo pipefail

ENDPOINT="${PIMP_AUDIO_MCP_ENDPOINT:-${ANVIL_MCP_ENDPOINT:-http://127.0.0.1:38765/mcp}}"
TOKEN="${PIMP_AUDIO_MCP_TOKEN:-${ANVIL_MCP_TOKEN:-}}"
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

grep_uri() {
  local uri="$1"
  local file="$2"
  local escaped="${uri//\//\\/}"
  grep -Fq "\"$uri\"" "$file" || grep -Fq "\"$escaped\"" "$file"
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
confirmation = structured.get("confirmation", {})
if confirmation.get("confirmArgument") != "confirm":
    sys.exit("submit_editor_inpaint dry-run did not advertise confirm argument")
if "elicitation/form" not in confirmation.get("supportedModes", []):
    sys.exit("submit_editor_inpaint dry-run did not advertise elicitation confirmation mode")
if request.get("mode") != "inpaint":
    sys.exit("submit_editor_inpaint dry-run did not resolve inpaint mode")
if request.get("regionPrompt") != "replace the selected phrase with a cleaner synth swell":
    sys.exit("submit_editor_inpaint dry-run did not echo regionPrompt")
'

python_check_editor_render_dry_run='
import json
import sys

data = json.load(sys.stdin)
structured = data.get("result", {}).get("structuredContent", {})
request = structured.get("request", {})
if structured.get("dryRun") is not True:
    sys.exit("render_editor_audio dry-run response did not set dryRun")
if structured.get("submitted") is not False:
    sys.exit("render_editor_audio dry-run unexpectedly submitted")
if structured.get("requiresLoadedAudio") is not True:
    sys.exit("render_editor_audio dry-run did not advertise loaded-audio requirement")
if request.get("format") != "wav":
    sys.exit("render_editor_audio dry-run did not resolve wav format")
if not str(request.get("outputPath", "")).endswith(".wav"):
    sys.exit("render_editor_audio dry-run did not resolve a wav output path")
'

python_check_editor_dsp_dry_run='
import json
import sys

data = json.load(sys.stdin)
structured = data.get("result", {}).get("structuredContent", {})
request = structured.get("request", {})
effects = request.get("effects", [])
if structured.get("dryRun") is not True:
    sys.exit("apply_editor_effect_chain dry-run response did not set dryRun")
if structured.get("submitted") is not False:
    sys.exit("apply_editor_effect_chain dry-run unexpectedly submitted")
if structured.get("requiresLoadedAudio") is not True:
    sys.exit("apply_editor_effect_chain dry-run did not advertise loaded-audio requirement")
if structured.get("requiresConfirmation") is not True:
    sys.exit("apply_editor_effect_chain dry-run did not advertise confirmation requirement")
confirmation = structured.get("confirmation", {})
if confirmation.get("confirmArgument") != "confirm":
    sys.exit("apply_editor_effect_chain dry-run did not advertise confirm argument")
if "elicitation/form" not in confirmation.get("supportedModes", []):
    sys.exit("apply_editor_effect_chain dry-run did not advertise elicitation confirmation mode")
if request.get("effectCount") != 2 or len(effects) != 2:
    sys.exit("apply_editor_effect_chain dry-run did not resolve the effect chain")
if effects[0].get("effect") != "fadeIn":
    sys.exit("apply_editor_effect_chain dry-run did not resolve fadeIn")
if effects[1].get("effect") != "normalize":
    sys.exit("apply_editor_effect_chain dry-run did not resolve normalize")
if effects[0].get("target", {}).get("type") != "selection":
    sys.exit("apply_editor_effect_chain dry-run did not resolve selection target")
'

python_check_release_memory_confirmation_required='
import json
import sys

data = json.load(sys.stdin)
result = data.get("result", {})
if result.get("isError") is not True:
    sys.exit("release_memory without confirmation was not rejected")
text = "\n".join(block.get("text", "") for block in result.get("content", []))
if "confirm: true" not in text:
    sys.exit("release_memory rejection did not mention confirm: true")
'

python_check_completion_contains='
import json
import sys

expected = sys.argv[1]
data = json.load(sys.stdin)
completion = data.get("result", {}).get("completion", {})
values = completion.get("values", [])
if expected not in values:
    sys.exit(f"completion response did not include {expected!r}: {values!r}")
if completion.get("total", 0) < 1:
    sys.exit("completion response did not report a positive total")
'

python_check_client_roots_unsupported='
import json
import sys

data = json.load(sys.stdin)
contents = data.get("result", {}).get("contents", [])
text = "\n".join(block.get("text", "") for block in contents)
payload = json.loads(text)
if payload.get("kind") != "clientRoots":
    sys.exit("client-roots resource did not return kind=clientRoots")
if payload.get("supported") is not False:
    sys.exit("smoke client should not advertise MCP roots")
if payload.get("count") != 0:
    sys.exit("client-roots resource returned roots for a roots-less client")
if payload.get("policy", {}).get("historyAudioPreferred") is not True:
    sys.exit("client-roots policy did not prefer history/audio ids")
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
    "clientInfo": { "name": "pimpaudio-smoke-mcp", "version": "1.0" }
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
grep -q '"apply_editor_effect_chain"' "$TMP_DIR/tools.json"
grep -q '"render_editor_audio"' "$TMP_DIR/tools.json"
grep -q '"list_operations"' "$TMP_DIR/tools.json"
grep -q '"get_operation"' "$TMP_DIR/tools.json"
grep -q '"cancel_operation"' "$TMP_DIR/tools.json"
grep -q '"release_memory"' "$TMP_DIR/tools.json"

echo "3a. release_memory confirmation guard"
rpc '{
  "jsonrpc": "2.0",
  "id": 30,
  "method": "tools/call",
  "params": {
    "name": "release_memory",
    "arguments": {}
  }
}' 30 > "$TMP_DIR/release-memory-confirmation-required.json"
/usr/bin/python3 -c "$python_check_release_memory_confirmation_required" < "$TMP_DIR/release-memory-confirmation-required.json"

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

echo "7. apply_editor_effect_chain dry-run"
rpc '{
  "jsonrpc": "2.0",
  "id": 6,
  "method": "tools/call",
  "params": {
    "name": "apply_editor_effect_chain",
    "arguments": {
      "effects": [
        { "effect": "fadeIn", "selectionStart": 0, "selectionEnd": 0.5 },
        { "effect": "normalize" }
      ],
      "dryRun": true
    }
  }
}' 6 > "$TMP_DIR/editor-dsp-dry-run.json"
/usr/bin/python3 -c "$python_check_editor_dsp_dry_run" < "$TMP_DIR/editor-dsp-dry-run.json"

echo "8. render_editor_audio dry-run"
rpc '{
  "jsonrpc": "2.0",
  "id": 7,
  "method": "tools/call",
  "params": {
    "name": "render_editor_audio",
    "arguments": {
      "format": "wav",
      "selectionStart": 0,
      "selectionEnd": 1,
      "dryRun": true
    }
  }
}' 7 > "$TMP_DIR/editor-render-dry-run.json"
/usr/bin/python3 -c "$python_check_editor_render_dry_run" < "$TMP_DIR/editor-render-dry-run.json"

echo "9. list_operations"
rpc '{
  "jsonrpc": "2.0",
  "id": 8,
  "method": "tools/call",
  "params": {
    "name": "list_operations",
    "arguments": { "limit": 5 }
  }
}' 8 > "$TMP_DIR/operations.json"
grep -q '"operations"' "$TMP_DIR/operations.json"

echo "10. resources/list"
rpc '{"jsonrpc":"2.0","id":9,"method":"resources/list"}' 9 > "$TMP_DIR/resources.json"
grep_uri "pimpaudio://history" "$TMP_DIR/resources.json"
grep_uri "pimpaudio://project/current" "$TMP_DIR/resources.json"
grep_uri "pimpaudio://track/current" "$TMP_DIR/resources.json"
grep_uri "pimpaudio://clip/current" "$TMP_DIR/resources.json"
grep_uri "pimpaudio://region/current-selection" "$TMP_DIR/resources.json"
grep_uri "pimpaudio://takes" "$TMP_DIR/resources.json"
grep_uri "pimpaudio://presets" "$TMP_DIR/resources.json"
grep_uri "pimpaudio://workflows" "$TMP_DIR/resources.json"
grep_uri "pimpaudio://editor" "$TMP_DIR/resources.json"
grep_uri "pimpaudio://operations" "$TMP_DIR/resources.json"
grep_uri "pimpaudio://client-roots" "$TMP_DIR/resources.json"

echo "10a. resources/read pimpaudio://client-roots"
rpc '{
  "jsonrpc": "2.0",
  "id": 90,
  "method": "resources/read",
  "params": { "uri": "pimpaudio://client-roots" }
}' 90 > "$TMP_DIR/client-roots.json"
/usr/bin/python3 -c "$python_check_client_roots_unsupported" < "$TMP_DIR/client-roots.json"

echo "11. prompts/list"
rpc '{"jsonrpc":"2.0","id":10,"method":"prompts/list"}' 10 > "$TMP_DIR/prompts.json"
grep -q '"improve_chorus"' "$TMP_DIR/prompts.json"
grep -q '"clean_up_region"' "$TMP_DIR/prompts.json"
grep -q '"production_ready_prompt"' "$TMP_DIR/prompts.json"

echo "12. prompts/get clean_up_region"
rpc '{
  "jsonrpc": "2.0",
  "id": 11,
  "method": "prompts/get",
  "params": {
    "name": "clean_up_region",
    "arguments": {
      "regionStart": "0",
      "regionEnd": "1",
      "issue": "remove a click"
    }
  }
}' 11 > "$TMP_DIR/prompt-clean-up-region.json"
grep -q 'apply_editor_effect_chain' "$TMP_DIR/prompt-clean-up-region.json"
grep -q 'submit_editor_inpaint' "$TMP_DIR/prompt-clean-up-region.json"

echo "12a. resources/templates/list"
rpc '{"jsonrpc":"2.0","id":110,"method":"resources/templates/list"}' 110 > "$TMP_DIR/resource-templates.json"
grep -q 'pimpaudio://workflow/{id}' "$TMP_DIR/resource-templates.json"
grep -q 'pimpaudio://operation/{id}' "$TMP_DIR/resource-templates.json"

echo "12b. completion/complete prompt issue"
rpc '{
  "jsonrpc": "2.0",
  "id": 111,
  "method": "completion/complete",
  "params": {
    "ref": { "type": "ref/prompt", "name": "clean_up_region" },
    "argument": { "name": "issue", "value": "cl" }
  }
}' 111 > "$TMP_DIR/completion-prompt-issue.json"
/usr/bin/python3 -c "$python_check_completion_contains" "click/pop cleanup" < "$TMP_DIR/completion-prompt-issue.json"

echo "12c. completion/complete workflow resource"
rpc '{
  "jsonrpc": "2.0",
  "id": 112,
  "method": "completion/complete",
  "params": {
    "ref": { "type": "ref/resource", "uri": "pimpaudio://workflow/{id}" },
    "argument": { "name": "id", "value": "clean" }
  }
}' 112 > "$TMP_DIR/completion-workflow-resource.json"
/usr/bin/python3 -c "$python_check_completion_contains" "clean_up_region" < "$TMP_DIR/completion-workflow-resource.json"

echo "13. list_history"
rpc '{
  "jsonrpc": "2.0",
  "id": 12,
  "method": "tools/call",
  "params": {
    "name": "list_history",
    "arguments": { "limit": 10 }
  }
}' 12 > "$TMP_DIR/history.json"

TRACE_IDS=()
while IFS= read -r trace_id; do
  if [[ -n "$trace_id" ]]; then
    TRACE_IDS+=("$trace_id")
  fi
done < <(/usr/bin/python3 -c "$python_get_trace_ids" < "$TMP_DIR/history.json")

AUDIO_REF="$(/usr/bin/python3 -c "$python_get_audio_ref" < "$TMP_DIR/history.json")"
if [[ -n "$AUDIO_REF" ]]; then
  echo "14. load_editor_audio dry-run"
  /usr/bin/python3 - "$AUDIO_REF" > "$TMP_DIR/load-editor-audio-request.json" <<'PY'
import json
import sys

ref = json.loads(sys.argv[1])
print(json.dumps({
    "jsonrpc": "2.0",
    "id": 13,
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
  rpc "$(cat "$TMP_DIR/load-editor-audio-request.json")" 13 > "$TMP_DIR/load-editor-audio-dry-run.json"
  grep -q '"dryRun"' "$TMP_DIR/load-editor-audio-dry-run.json"
else
  echo "14. skipped load_editor_audio dry-run; no existing history audio file found"
fi

if [[ "${#TRACE_IDS[@]}" -ge 1 ]]; then
  FIRST_ID="${TRACE_IDS[0]}"

  echo "15. get_trace ($FIRST_ID)"
  rpc "{
    \"jsonrpc\": \"2.0\",
    \"id\": 14,
    \"method\": \"tools/call\",
    \"params\": {
      \"name\": \"get_trace\",
      \"arguments\": { \"id\": \"$FIRST_ID\" }
    }
  }" 14 >/dev/null

  echo "16. resources/read pimpaudio://history/$FIRST_ID"
  rpc "{
    \"jsonrpc\": \"2.0\",
    \"id\": 15,
    \"method\": \"resources/read\",
    \"params\": { \"uri\": \"pimpaudio://history/$FIRST_ID\" }
  }" 15 > "$TMP_DIR/history-resource.json"
  grep -q '"contents"' "$TMP_DIR/history-resource.json"

  echo "17. resources/read pimpaudio://trace/$FIRST_ID"
  rpc "{
    \"jsonrpc\": \"2.0\",
    \"id\": 16,
    \"method\": \"resources/read\",
    \"params\": { \"uri\": \"pimpaudio://trace/$FIRST_ID\" }
  }" 16 > "$TMP_DIR/trace-resource.json"
  grep -q '"contents"' "$TMP_DIR/trace-resource.json"
else
  echo "15-17. skipped trace checks; no traced history rows found"
fi

if [[ "${#TRACE_IDS[@]}" -ge 2 ]]; then
  FIRST_ID="${TRACE_IDS[0]}"
  SECOND_ID="${TRACE_IDS[1]}"

  echo "18. compare_traces ($FIRST_ID vs $SECOND_ID)"
  rpc "{
    \"jsonrpc\": \"2.0\",
    \"id\": 17,
    \"method\": \"tools/call\",
    \"params\": {
      \"name\": \"compare_traces\",
      \"arguments\": {
        \"left\": \"$FIRST_ID\",
        \"right\": \"$SECOND_ID\"
      }
    }
  }" 17 >/dev/null
else
  echo "18. skipped compare; fewer than two traced history rows found"
fi

echo "19. legacy anvil.list_history alias"
rpc '{
  "jsonrpc": "2.0",
  "id": 18,
  "method": "tools/call",
  "params": {
    "name": "anvil.list_history",
    "arguments": { "limit": 1 }
  }
}' 18 >/dev/null

echo "20. legacy anvil_list_history alias"
rpc '{
  "jsonrpc": "2.0",
  "id": 19,
  "method": "tools/call",
  "params": {
    "name": "anvil_list_history",
    "arguments": { "limit": 1 }
  }
}' 19 >/dev/null

echo "21. GET /mcp stream + progress notification"
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
  "id": 20,
  "method": "tools/call",
  "params": {
    "_meta": { "progressToken": "pimpaudio-smoke-progress" },
    "name": "generation_status",
    "arguments": {}
  }
}' 20 >/dev/null

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
grep -q 'pimpaudio-smoke-progress' "$TMP_DIR/get-mcp.sse"

echo "MCP smoke test passed."
