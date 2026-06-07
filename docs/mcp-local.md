# Local MCP Server

PiMP Audio Studio can expose a localhost-only MCP endpoint for local agents and scripts.
The server is off by default and is intended for same-machine workflows.

The current implementation uses the official Swift MCP SDK for lifecycle,
tool/resource dispatch, structured tool output, and stateful Streamable HTTP
session handling. PiMP Audio Studio still owns the localhost listener and optional token
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

PiMP Audio Studio speaks MCP Streamable HTTP through the Swift SDK's stateful server
transport:

- `POST /mcp` accepts JSON-RPC requests and can return `text/event-stream`
  responses.
- `initialize` returns an `MCP-Session-Id` header. Clients must echo that
  session id on later `POST`, `GET`, and `DELETE` requests.
- `GET /mcp` opens the standalone SSE stream used for server-initiated
  messages, including `notifications/progress`.
- `DELETE /mcp` terminates the session. PiMP Audio Studio resets the embedded SDK
  transport afterward so another local client can initialize without an app
  restart.
- Raw curl callers should parse SSE `data:` events as JSON-RPC messages.
  `scripts/smoke-pimpaudio-mcp.sh` is the canonical shell example.

## Tools

The local MCP endpoint exposes read/status tools:

- `list_models`
- `list_adapters`
- `training_status`
- `list_training_datasets`
- `list_training_runs`
- `tail_training_log`
- `list_history`
- `get_trace`
- `compare_traces`
- `enhance_prompt`
- `write_lyrics`
- `generation_status`
- `editor_status`
- `load_editor_audio`
- `submit_editor_inpaint`
- `apply_editor_effect_chain`
- `render_editor_audio`
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

`training_status` returns the current local LoRA/LoKR training state, active
progress metrics, active run-log summary, and AUDIODIAG paths. `list_training_datasets`
returns visible local dataset packages from the current output folder plus the
app support fallback, `list_training_runs` returns installed adapters with
`training_log.json` summaries, and `tail_training_log` returns recent lines from
the active or latest Diagnostics log filtered to training lines by default.
These tools are read-only; MCP does not start local training yet.

`editor_status` returns editor readiness, loaded audio metadata, supported
editor action descriptors, and the active editor operation when present.
`load_editor_audio` loads a PiMP Audio Studio history/audio id, or a readable local audio
path, into the shared editor coordinator as a background operation. It returns
an `operationID` for polling or cancellation before inpaint/DSP/render tools
mutate the editor buffer. History/audio ids are preferred because
sandboxed app builds may not have permission to read arbitrary paths.
`submit_editor_inpaint` submits a selected-region editor mutation against the
loaded editor audio. Use `dryRun: true` to validate and echo the resolved
request without touching audio. Actual runs require loaded editor audio and
either `confirm: true` or MCP form elicitation approval, then return an
`editor.inpaint` `operationID` for polling or cancellation.
`apply_editor_effect_chain` submits one or more DSP effects against the loaded
editor audio. It accepts a single `effect` or an ordered `effects` array,
optional `selectionStart`/`selectionEnd`, per-effect `parameters`, and requires
`confirm: true` or MCP form elicitation approval for actual mutations.
Supported effects are
`trimToSelection`, `deleteSelection`, `fadeIn`, `fadeOut`, `normalize`, `eq`,
`reverb`, `timeStretch`, and `pitchShift`; the operation reports as
`editor.applyEffectChain`.
`render_editor_audio` writes the current editor buffer, or an optional selected
time range, to a WAV/AIFF file. It defaults to an `MCPExports` folder beside
the loaded source audio and returns an `editor.renderAudio` `operationID`.

`list_operations`, `get_operation`, and `cancel_operation` expose the current
operation layer for long-running work. Generation submissions return an
`operationID`; editor audio loads return `editor.loadAudio` operations,
editor inpaint/repaint submissions return `editor.inpaint`, DSP chains return
`editor.applyEffectChain`, and editor render submissions return
`editor.renderAudio`. Clients can poll those ids for status/progress or cancel
them through the same operation registry used by the app.

Protocol progress is currently request-scoped: when a client includes
`_meta.progressToken` on `tools/call`, PiMP Audio Studio emits start/completed/failed
`notifications/progress` messages on the standalone `GET /mcp` stream.
If a generation tool submits background work, the tool request's progress
notification completes after the operation is accepted and includes the
`operationID`. Background generation/editor progress then uses the operation
record.
MCP permits progress tokens to stay valid after the original response only for
task-augmented requests, and the Swift SDK version currently integrated here
does not expose task capabilities yet.

For guarded mutations, PiMP Audio Studio uses the standard MCP `elicitation/create` flow
when the initialized client advertises form elicitation support and the tool
call omits `confirm: true`. The elicitation asks for an `approved` boolean;
only `action: "accept"` with `approved: true` is treated as confirmation.
Decline/cancel responses return a no-op structured result. Clients without
elicitation support still use the compatibility fallback: call the tool again
with `confirm: true` after explicit user approval.

It also exposes controlled app actions:

- `mark_preferred_take`
- `apply_adapter`
- `generate_song`
- `generate`
- `load_editor_audio`
- `submit_editor_inpaint`
- `apply_editor_effect_chain`
- `render_editor_audio`

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

`submit_editor_inpaint` submits a confirmed editor inpaint/repaint action. It
accepts `regionPrompt`, a selected time range (`selectionStart`/`selectionEnd`
or a `selection` object), optional frequency band bounds, strength/guidance
controls, pipeline choice, and seed. It is intentionally confirmation-gated
because it mutates the shared editor buffer.

`apply_editor_effect_chain` applies confirmed DSP edits to the shared editor
buffer and refreshes editable latents so future inpaint/render calls see the
edited audio. Use `effect` for a single effect or `effects` for an ordered
chain. Selection-only effects require a selected time range; other effects
default to the whole track.

`render_editor_audio` writes the current editor audio to disk. It accepts an
optional selected time range, `format: "wav"` or `"aiff"`, `outputPath` or
`outputDirectory`/`fileName`, and `overwrite: true` when replacing an existing
file is intentional. `dryRun: true` resolves the output path without writing.
For sandboxed app builds, prefer the default `MCPExports` path beside the
loaded source audio or an app-accessible output directory; arbitrary paths can
be rejected by macOS even when the shell can write there.

It exposes one guarded cleanup action:

- `release_memory`

`release_memory` runs the same release path as **Settings > Models >
Release GPU Memory**: it cancels active generation if needed, unloads model
weights, evicts the local LLM, and flushes the MLX GPU cache. It requires
`confirm: true` or MCP form elicitation approval and returns before/after
memory counters.

Legacy `anvil.` and `anvil_` tool prefixes are no longer part of the public MCP
surface. Use the plain snake_case tool names listed above.

## Resources

The same data is also available as MCP resources:

- `pimpaudio://models`
- `pimpaudio://model/{id}`
- `pimpaudio://adapters`
- `pimpaudio://adapter/{id}`
- `pimpaudio://training`
- `pimpaudio://training-datasets`
- `pimpaudio://training-runs`
- `pimpaudio://history`
- `pimpaudio://history/{id}`
- `pimpaudio://project/current`
- `pimpaudio://track/current`
- `pimpaudio://clip/current`
- `pimpaudio://region/current-selection`
- `pimpaudio://takes`
- `pimpaudio://take/{id}`
- `pimpaudio://presets`
- `pimpaudio://preset/{id}`
- `pimpaudio://workflows`
- `pimpaudio://workflow/{id}`
- `pimpaudio://trace/{id}`
- `pimpaudio://audio/{id}`
- `pimpaudio://editor`
- `pimpaudio://operations`
- `pimpaudio://operation/{id}`
- `pimpaudio://client-roots`

Audio resources return local file metadata and paths only. They do not stream
audio bytes over MCP.

`pimpaudio://client-roots` reports whether the initialized MCP client advertised
the MCP roots capability, the latest roots returned by `roots/list`, and
PiMP Audio Studio's local path policy. Roots are advisory filesystem boundaries from the
client; they do not bypass macOS sandboxing or grant PiMP Audio Studio new file
permissions. For editor loading, history/audio ids are still preferred. For
renders, the default `MCPExports` folder beside the loaded source audio remains
the safest path unless the user intentionally chooses a client-advertised root.

`pimpaudio://project/current` is the top-level local workstation snapshot. It links
the current editor track, clip, current/last region, recent takes, presets,
workflow prompts, editor status, and operation registry.

`pimpaudio://takes` and `pimpaudio://take/{id}` present generation history in a
workstation-friendly shape with links to the matching history, trace, and audio
resources.

The endpoint also supports `resources/templates/list` for parameterized
resource discovery. Current templates are:

- `pimpaudio://model/{id}`
- `pimpaudio://adapter/{id}`
- `pimpaudio://history/{id}`
- `pimpaudio://take/{id}`
- `pimpaudio://trace/{id}`
- `pimpaudio://audio/{id}`
- `pimpaudio://preset/{id}`
- `pimpaudio://workflow/{id}`
- `pimpaudio://operation/{id}`

`pimpaudio://presets` includes saved generation presets plus built-in guidance,
inpaint, reverb, and DSP effect catalogs. Individual saved presets are
addressable as `pimpaudio://preset/{id}`.

`pimpaudio://workflows` and `pimpaudio://workflow/{id}` expose the same workflow prompt
catalog that is available through MCP prompts, including required arguments,
recommended resources, and relevant tools.

## Prompts

The local MCP endpoint advertises reusable workflow prompts through the SDK
`prompts/list` and `prompts/get` methods:

- `improve_chorus`
- `create_mix_variants`
- `clean_up_region`
- `prepare_stems_for_review`
- `production_ready_prompt`

These prompts do not mutate app state by themselves. They give a client a
structured workflow recipe that points at PiMP Audio Studio resources and tools, with the
same dry-run and guarded-confirmation expectations used by the direct tools.

PiMP Audio Studio advertises MCP completions through `completion/complete`. Completions
cover workflow prompt arguments such as `takeID`, `regionStart`, `regionEnd`,
`variantCount`, `duration`, `direction`, `issue`, and `outputDirectory`, plus
resource-template ids for models, adapters, history/takes/traces/audio,
presets, workflows, and operations. When a client advertises roots and returns
file roots through `roots/list`, `outputDirectory` completion can also suggest
those root paths.

## Smoke Test

The repo includes a smoke script:

```bash
scripts/smoke-pimpaudio-mcp.sh
```

If token mode is enabled:

```bash
PIMP_AUDIO_MCP_TOKEN='<token>' scripts/smoke-pimpaudio-mcp.sh
```

The smoke script initializes a session, parses POST SSE responses, checks the
tool/resource/prompt catalog, verifies editor and operation resources, dry-runs
editor audio loading when a history WAV is available, opens `GET /mcp`, and
checks a `notifications/progress` event.

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
- Training MCP tools are read-only diagnostics and dataset/run discovery; they
  do not start training, cancel training, or upload audio.
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
  background editor load, inpaint, DSP, and render tasks. Client
  `notifications/cancelled` apply to in-flight MCP requests through the SDK;
  after background work has been submitted, use `cancel_operation` for the app
  operation itself.
- `submit_editor_inpaint`, `apply_editor_effect_chain`, and `release_memory`
  are intentionally guarded. They accept `confirm: true` as the compatibility
  path and use MCP form elicitation when the client advertises support.
- The standards direction is tracked in `docs/mcp-standards-roadmap.md`.
