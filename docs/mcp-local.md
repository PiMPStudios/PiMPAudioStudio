# Local MCP Server

Anvil can expose a localhost-only MCP endpoint for local agents and scripts.
The server is off by default and is intended for same-machine workflows.

The current implementation uses the official Swift MCP SDK for lifecycle,
tool/resource dispatch, structured tool output, and stateful Streamable HTTP
session handling. Anvil still owns the localhost listener and optional token
gate so it can stay embedded inside the desktop app.

## Enable

1. Open **Settings > MCP**.
2. Turn on **Enable MCP Server**.
3. Copy the endpoint shown in Settings.

Default endpoint:

```text
http://127.0.0.1:38765/mcp
```

The optional bearer token is disabled by default for localhost-only use. If you
enable it, clients must send:

```text
Authorization: Bearer <token>
```

## Transport Behavior

Anvil speaks MCP Streamable HTTP through the Swift SDK's stateful server
transport:

- `POST /mcp` accepts JSON-RPC requests and can return `text/event-stream`
  responses.
- `initialize` returns an `MCP-Session-Id` header. Clients must echo that
  session id on later `POST`, `GET`, and `DELETE` requests.
- `GET /mcp` opens the standalone SSE stream used for server-initiated
  messages, including `notifications/progress`.
- `DELETE /mcp` terminates the session. Anvil resets the embedded SDK
  transport afterward so another local client can initialize without an app
  restart.
- Raw curl callers should parse SSE `data:` events as JSON-RPC messages.
  `scripts/smoke-mcp.sh` is the canonical shell example.

## Tools

The local MCP endpoint exposes read/status tools:

- `list_models`
- `list_adapters`
- `list_history`
- `get_trace`
- `compare_traces`
- `enhance_prompt`
- `write_lyrics`
- `generation_status`
- `editor_status`
- `load_editor_audio`
- `list_operations`
- `get_operation`
- `cancel_operation`

`enhance_prompt` and `write_lyrics` call the same local/API
intelligence services used by the Generate view. They return text only and do
not update the current prompt, lyric field, history, traces, models, or audio.

`generation_status` returns the current generation state and latest
completed result metadata, including local audio/trace paths when available.
When a generation was submitted through MCP, it also returns the current
operation record.

`editor_status` returns editor readiness, loaded audio metadata, supported
editor action descriptors, and the active editor operation when present.
`load_editor_audio` loads an Anvil history/audio id, or a readable local audio
path, into the shared editor coordinator as a background operation. It returns
an `operationID` for polling or cancellation before future inpaint/DSP/render
tools mutate the editor buffer. History/audio ids are preferred because
sandboxed app builds may not have permission to read arbitrary paths.

`list_operations`, `get_operation`, and `cancel_operation` expose the current
operation layer for long-running work. Generation submissions return an
`operationID`; editor audio loads now return `editor.loadAudio` operations.
Clients can poll those ids for status/progress or cancel them through the same
operation registry used by the app.

Protocol progress is currently request-scoped: when a client includes
`_meta.progressToken` on `tools/call`, Anvil emits start/completed/failed
`notifications/progress` messages on the standalone `GET /mcp` stream.
If a generation tool submits background work, the tool request's progress
notification completes after the operation is accepted and includes the
`operationID`. Background generation/editor progress then uses the operation
record.
MCP permits progress tokens to stay valid after the original response only for
task-augmented requests, and the Swift SDK version currently integrated here
does not expose task capabilities yet.

It also exposes controlled app actions:

- `mark_preferred_take`
- `apply_adapter`
- `generate_song`
- `generate`
- `load_editor_audio`

`mark_preferred_take` only updates review metadata on an existing trace:
the preferred flag and optional listening note. It mirrors the same values into
`history.json`, matching the Generation Analyzer's `Mark Preferred` / notes
behavior. It does not modify audio files, models, adapters, tensors, or
generation settings.

`apply_adapter` selects, stacks, or clears ACE-Step LoRA adapters for the
next generation. It validates that adapters are installed and ACE-Step
compatible, updates the same generation-service state used by the UI, persists
the primary adapter, and notifies the Generate view when it is open.

`generate_song` can optionally enhance the prompt, write lyrics, apply
adapters, then submit generation through the same service path as the Generate
button.

`generate` submits a generation request directly. It returns immediately; use
`generation_status`, `list_history`, or trace resources to observe completion. Pass
`dryRun: true` to validate and echo the resolved request without starting a
render.

`load_editor_audio` submits a background editor load. It validates the source
up front, returns an `operationID`, and only updates the editor's loaded-audio
state after the coordinator has encoded editable latents.

It exposes one guarded cleanup action:

- `release_memory`

`release_memory` runs the same release path as **Settings > Models >
Release GPU Memory**: it cancels active generation if needed, unloads model
weights, evicts the local LLM, and flushes the MLX GPU cache. It requires
`confirm: true` and returns before/after memory counters.

Legacy aliases using `anvil.` and `anvil_` prefixes are still accepted for
compatibility, for example `anvil.list_history` and `anvil_list_history`.

## Resources

The same data is also available as MCP resources:

- `anvil://models`
- `anvil://model/{id}`
- `anvil://adapters`
- `anvil://adapter/{id}`
- `anvil://history`
- `anvil://history/{id}`
- `anvil://trace/{id}`
- `anvil://audio/{id}`
- `anvil://editor`
- `anvil://operations`
- `anvil://operation/{id}`

Audio resources return local file metadata and paths only. They do not stream
audio bytes over MCP.

## Smoke Test

The repo includes a smoke script:

```bash
scripts/smoke-mcp.sh
```

If token mode is enabled:

```bash
ANVIL_MCP_TOKEN='<token>' scripts/smoke-mcp.sh
```

The smoke script initializes a session, parses POST SSE responses, checks the
tool/resource catalog, verifies editor and operation resources, dry-runs editor
audio loading when a history WAV is available, opens `GET /mcp`, and checks a
`notifications/progress` event.

Manual curl testing is possible, but every request after initialize must send
the returned `MCP-Session-Id`. The raw response body may be SSE, so inspect
`data:` lines or reuse the smoke script's parser.

Initialize and capture the session:

```bash
ENDPOINT='http://127.0.0.1:38765/mcp'
HEADERS="$(mktemp)"

curl -s -D "$HEADERS" "$ENDPOINT" \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -H 'MCP-Protocol-Version: 2025-06-18' \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
      "protocolVersion": "2025-06-18",
      "capabilities": {},
      "clientInfo": { "name": "curl", "version": "1.0" }
    }
  }'

SESSION_ID="$(
  awk 'tolower($1)=="mcp-session-id:" { gsub("\r", "", $2); print $2 }' "$HEADERS"
)"
```

List tools:

```bash
curl -s "$ENDPOINT" \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -H 'MCP-Protocol-Version: 2025-06-18' \
  -H "MCP-Session-Id: $SESSION_ID" \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/list"}'
```

Call a tool:

```bash
curl -s "$ENDPOINT" \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -H 'MCP-Protocol-Version: 2025-06-18' \
  -H "MCP-Session-Id: $SESSION_ID" \
  -d '{
    "jsonrpc": "2.0",
    "id": 3,
    "method": "tools/call",
    "params": {
      "name": "list_history",
      "arguments": { "limit": 10 }
    }
  }'
```

Open the standalone stream before calling a tool with a progress token:

```bash
curl -s --no-buffer "$ENDPOINT" \
  -H 'Accept: text/event-stream' \
  -H 'MCP-Protocol-Version: 2025-06-18' \
  -H "MCP-Session-Id: $SESSION_ID"
```

Then call a progress-aware tool request from another shell:

```bash
curl -s "$ENDPOINT" \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -H 'MCP-Protocol-Version: 2025-06-18' \
  -H "MCP-Session-Id: $SESSION_ID" \
  -d '{
    "jsonrpc": "2.0",
    "id": 4,
    "method": "tools/call",
    "params": {
      "_meta": { "progressToken": "status-check" },
      "name": "generation_status",
      "arguments": {}
    }
  }'
```

Terminate the session:

```bash
curl -s -X DELETE "$ENDPOINT" \
  -H 'Accept: application/json, text/event-stream' \
  -H 'MCP-Protocol-Version: 2025-06-18' \
  -H "MCP-Session-Id: $SESSION_ID"
```

## Notes

- The server binds to `127.0.0.1` only.
- It does not expose raw tensors, model weights, or remote/LAN access.
- Destructive and expensive actions are intentionally deferred.
- `compare_traces` uses the same Model, Adapters, Sampling, Prompt &
  Lyrics, and Audio grouping as the Generation Analyzer compare view.
- `enhance_prompt` and `write_lyrics` may load/use the configured
  local LLM or API lyric backends, but their responses are return-only.
- `mark_preferred_take` is intentionally scoped to trace/history review
  metadata only.
- `apply_adapter` changes app adapter-selection state but does not start
  generation, load model weights, or alter adapter files.
- `generate` is non-blocking. It starts the same app generation path and
  returns immediately so MCP clients can poll status or history instead of
  holding a long-running HTTP request open.
- MCP-submitted generation now creates an operation record with `progress`,
  `total`, and `message` fields normalized to the whole operation lifetime.
  Stage-local counters, when available, are exposed as `stageProgress` and
  `stageTotal`.
- Request-scoped `notifications/progress` are available when a client sends
  `_meta.progressToken` and keeps `GET /mcp` open. Full background-operation
  progress notifications should be implemented through MCP task-augmented
  requests once the SDK supports task capabilities.
- `load_editor_audio` is the first editor operation. It accepts `id`,
  `historyID`, `audioID`, or a readable `path`; registers an `editor.loadAudio`
  record; reads the file; encodes editor latents; and updates
  `editor_status.loadedAudio` when complete.
- `cancel_operation` currently supports active generation operations and
  background editor load tasks. Client `notifications/cancelled` apply to
  in-flight MCP requests through the SDK; after background work has been
  submitted, use `cancel_operation` for the app operation itself. Future
  editor/render operations should register with the same operation layer before
  they are exposed as mutating MCP tools.
- `release_memory` is intentionally guarded because it cancels active
  generation and unloads resident model state. The next generation or LLM tool
  call reloads what it needs.
- The standards direction is tracked in `docs/mcp-standards-roadmap.md`.
