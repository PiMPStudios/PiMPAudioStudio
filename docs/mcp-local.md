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
3. Optionally choose bind address (`127.0.0.1` or `0.0.0.0`) and port.
4. Copy the endpoint shown in Settings.

Default bind and endpoint:

```text
Bind: 127.0.0.1:38765
http://127.0.0.1:38765/mcp
```

Bind address:

- `127.0.0.1` — localhost only (default). The listener is **IPv4-only**;
  clients must use `127.0.0.1`, not `localhost` if that resolves to `::1`.
- `0.0.0.0` — all IPv4 interfaces (LAN reachable). Switching to this mode
  **requires** a bearer token; it cannot be turned off until you bind localhost
  again. Clients on other machines should use the Mac's LAN IP (Settings shows a
  network endpoint when one can be detected), not the literal string `0.0.0.0`.
  CORS echoes a trusted `Origin` instead of `*`.

Port:

- Default `38765`.
- Allowed range `1024`–`65535`.
- Apply a new port from Settings; the server restarts when enabled.

Override the smoke script target with:

```bash
PIMP_AUDIO_MCP_ENDPOINT='http://127.0.0.1:38765/mcp' scripts/smoke-pimpaudio-mcp.sh
```

The optional bearer token is disabled by default for localhost-only use. If you
enable it, or bind to all interfaces (which forces the token on), clients must
send:

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
- If a local client abandons a session without a clean shutdown, the next
  `initialize` also rebuilds the embedded SDK transport instead of requiring an
  app restart.
- Raw curl callers should parse SSE `data:` events as JSON-RPC messages.
  `scripts/smoke-pimpaudio-mcp.sh` is the canonical shell example.

## Tools

The local MCP endpoint exposes these tools:

### Read / inspect

- `list_models`
- `list_adapters`
- `list_history`
- `get_trace`
- `compare_traces`
- `training_status`
- `list_training_datasets`
- `list_training_runs`
- `inspect_training_dataset`
- `inspect_training_run`
- `compare_training_runs`
- `tail_training_log`
- `tail_diagnostics_log`
- `generation_status`
- `editor_status`
- `list_operations`
- `get_operation`
- `runpod_status`
- `remote_training_job_status`

### Intelligence

- `enhance_prompt`
- `write_lyrics`

### Generation

- `apply_adapter`
- `generate`
- `generate_song`
- `mark_preferred_take`
- `cancel_generation` (confirm)
- `run_adapter_audition`

### Editor

- `load_editor_audio`
- `submit_editor_inpaint`
- `apply_editor_effect_chain`
- `render_editor_audio`

### Local training

- `training_preflight`
- `start_training`
- `stop_training`

### Datasets

- `build_training_dataset`
- `enrich_training_dataset`
- `update_training_dataset_review`

### Adapters / models / library

- `import_adapter`
- `delete_adapter`
- `merge_adapters`
- `download_model`
- `set_active_model`
- `delete_history_entries`
- `save_generation_preset`
- `delete_generation_preset`
- `set_verbose_diagnostics`

### Remote GPU

- `refresh_runpod_launch_options`
- `launch_runpod_pod` (supports optional `region` / `dataCenterIDs`)
- `stop_runpod_pod`
- `create_remote_training_job`
- `upload_remote_training_job`
- `start_remote_training_job`
- `pull_remote_training_job`

### Operations / cleanup

- `cancel_operation` (confirm)
- `release_memory`

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
`training_log.json` summaries, `inspect_training_dataset` returns package
validation, annotation coverage, sample rows, and dataset-size-aware LoRA/LoKR
recommendations, `inspect_training_run` returns adapter config, progress/event
tails, and health metrics, `compare_training_runs` returns a compact comparison
table for recent or selected adapter runs, and `tail_training_log` returns
recent lines from the active or latest Diagnostics log filtered to training
lines by default.

`training_preflight` resolves a reviewed local dataset and LoRA/LoKR
configuration without starting training. It accepts the same practical knobs as
the Training sheet: `kind`, `recipe`, rank/linear dim, alpha/alphaScale,
dropout, epochs, layer coverage, projection coverage, target layers, LR,
gradient accumulation, and dataset id/path aliases. It returns the same local
memory/config guard report used by TrainingService, including known blocked
shapes such as broad full-attention LoKR on XL models.

`start_training` uses the same resolver but defaults to `dryRun: true`. To
actually start local training, pass `dryRun: false` and either approve the MCP
elicitation form or pass `confirm: true` after explicit approval. `stop_training`
uses the TrainingService stop path and can save a checkpoint when possible via
`saveCheckpoint: true`; it is also guarded by MCP elicitation or `confirm: true`.

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
operation layer for long-running work. `cancel_operation` requires `confirm:
true` or MCP form elicitation, same as `cancel_generation`. Generation
submissions return an
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
Decline/cancel responses return a no-op structured result. If elicitation
throws, the tool is **not** run. Clients without elicitation support must call
the tool again with `confirm: true` after explicit user approval.

It also exposes controlled app actions:

- `mark_preferred_take`
- `apply_adapter`
- `generate_song`
- `generate`
- `start_training`
- `stop_training`
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
button. It defaults to `dryRun: true` (and skips lyric API calls / adapter
mutation). Actual runs need `dryRun: false` plus `confirm: true` or MCP form
approval. When `clearAdapter: true` is combined with `adapter` or `adapters`,
the orchestrator clears stale selection first and then applies the requested
adapter selection before submission.

`generate` submits a generation request directly and defaults to `dryRun: true`.
Actual runs need `dryRun: false` plus `confirm: true` or MCP form approval. It
returns immediately; use `generation_status`, `list_history`, or trace resources
to observe completion.

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

### Remote GPU lifecycle tools

The managed RunPod job path is also exposed for agents:

1. `create_remote_training_job` — export a reviewed package + `remote_job.json` zip (`dryRun` default).
2. `launch_runpod_pod` — create Secure Cloud capacity (`dryRun` default; optional `region`).
3. `upload_remote_training_job` — SCP job zip + portable worker (confirm).
4. `start_remote_training_job` — start detached setup/training (confirm).
5. `remote_training_job_status` — poll phase/log tail (read-only).
6. `pull_remote_training_job` — download `remote_artifacts` and import the adapter (confirm).
7. `stop_runpod_pod` — terminate/delete the pod (confirm).

SSH private keys and RunPod API keys remain UI/Keychain-managed (not MCP-writable).
Pass `sshKeyPath` only when the app already has a readable key, or after the user
configured one in **Build → Training → Remote GPU**.


## Resources

The same data is also available as MCP resources:

- `pimpaudio://models`
- `pimpaudio://model/{id}`
- `pimpaudio://adapters`
- `pimpaudio://adapter/{id}`
- `pimpaudio://training`
- `pimpaudio://training-datasets`
- `pimpaudio://training-dataset/{id}`
- `pimpaudio://training-runs`
- `pimpaudio://training-run/{id}`
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
checks a `notifications/progress` event. It then deletes the session and
verifies that a fresh `initialize` succeeds.

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
  `write_lyrics` requires confirmation because it may call a paid API.
- `mark_preferred_take` is intentionally scoped to trace/history review
  metadata only and requires confirmation.
- `apply_adapter` changes app adapter-selection state but does not start
  generation, load model weights, or alter adapter files. It requires
  confirmation.
- `generate` is non-blocking and defaults to dry-run. A confirmed submit
  starts the same app generation path and returns immediately so MCP clients
  can poll status or history instead of holding a long-running HTTP request
  open.
- MCP-submitted generation now creates an operation record with `progress`,
  `total`, and `message` fields normalized to the whole operation lifetime.
  Stage-local counters, when available, are exposed as `stageProgress` and
  `stageTotal`.
- Request-scoped `notifications/progress` are available when a client sends
  `_meta.progressToken` and keeps `GET /mcp` open. Full background-operation
  progress notifications should be implemented through MCP task-augmented
  requests once the SDK supports task capabilities.
- `load_editor_audio` is the first editor operation and defaults to dry-run.
  A confirmed load accepts `id`, `historyID`, `audioID`, or a readable `path`;
  registers an `editor.loadAudio` record; reads the file; encodes editor
  latents; and updates `editor_status.loadedAudio` when complete.
- `cancel_operation` currently supports active generation operations and
  background editor load, inpaint, DSP, and render tasks. It requires
  `confirm: true` or MCP form elicitation. Client
  `notifications/cancelled` apply to in-flight MCP requests through the SDK;
  after background work has been submitted, use `cancel_operation` for the app
  operation itself.
- `submit_editor_inpaint`, `apply_editor_effect_chain`, and `release_memory`
  are intentionally guarded. They accept `confirm: true` as the compatibility
  path and use MCP form elicitation when the client advertises support.
- The standards direction is tracked in `docs/mcp-standards-roadmap.md`.
