# Local MCP Server

Anvil can expose a localhost-only MCP endpoint for local agents and scripts.
The server is off by default and is intended for same-machine workflows.

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

## Tools

The local MCP endpoint exposes read tools:

- `anvil.list_models`
- `anvil.list_adapters`
- `anvil.list_history`
- `anvil.get_trace`
- `anvil.compare_traces`
- `anvil.enhance_prompt`
- `anvil.write_lyrics`
- `anvil.generation_status`

`anvil.enhance_prompt` and `anvil.write_lyrics` call the same local/API
intelligence services used by the Generate view. They return text only and do
not update the current prompt, lyric field, history, traces, models, or audio.

`anvil.generation_status` returns the current generation state and latest
completed result metadata, including local audio/trace paths when available.

It also exposes one controlled metadata write:

- `anvil.mark_preferred_take`
- `anvil.apply_adapter`
- `anvil.generate`

`anvil.mark_preferred_take` only updates review metadata on an existing trace:
the preferred flag and optional listening note. It mirrors the same values into
`history.json`, matching the Generation Analyzer's `Mark Preferred` / notes
behavior. It does not modify audio files, models, adapters, tensors, or
generation settings.

`anvil.apply_adapter` selects, stacks, or clears ACE-Step LoRA adapters for the
next generation. It validates that adapters are installed and ACE-Step
compatible, updates the same generation-service state used by the UI, persists
the primary adapter, and notifies the Generate view when it is open.

`anvil.generate` submits a generation request through the same service path as
the Generate button. It returns immediately; use `anvil.generation_status`,
`anvil.list_history`, or trace resources to observe completion. Pass
`dryRun: true` to validate and echo the resolved request without starting a
render.

It exposes one guarded cleanup action:

- `anvil.release_memory`

`anvil.release_memory` runs the same release path as **Settings > Models >
Release GPU Memory**: it cancels active generation if needed, unloads model
weights, evicts the local LLM, and flushes the MLX GPU cache. It requires
`confirm: true` and returns before/after memory counters.

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

Audio resources return local file metadata and paths only. They do not stream
audio bytes over MCP.

## Smoke Test With Curl

The repo includes a smoke script:

```bash
scripts/smoke-mcp.sh
```

If token mode is enabled:

```bash
ANVIL_MCP_TOKEN='<token>' scripts/smoke-mcp.sh
```

You can also run the calls manually.

Initialize:

```bash
curl -s http://127.0.0.1:38765/mcp \
  -H 'Content-Type: application/json' \
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
```

List tools:

```bash
curl -s http://127.0.0.1:38765/mcp \
  -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/list"}'
```

Call `anvil.list_history`:

```bash
curl -s http://127.0.0.1:38765/mcp \
  -H 'Content-Type: application/json' \
  -d '{
    "jsonrpc": "2.0",
    "id": 3,
    "method": "tools/call",
    "params": {
      "name": "anvil.list_history",
      "arguments": { "limit": 10 }
    }
  }'
```

Enhance a prompt:

```bash
curl -s http://127.0.0.1:38765/mcp \
  -H 'Content-Type: application/json' \
  -d '{
    "jsonrpc": "2.0",
    "id": 4,
    "method": "tools/call",
    "params": {
      "name": "anvil.enhance_prompt",
      "arguments": {
        "prompt": "slow dark guitar riff",
        "mode": "music"
      }
    }
  }'
```

Write lyrics:

```bash
curl -s http://127.0.0.1:38765/mcp \
  -H 'Content-Type: application/json' \
  -d '{
    "jsonrpc": "2.0",
    "id": 5,
    "method": "tools/call",
    "params": {
      "name": "anvil.write_lyrics",
      "arguments": {
        "prompt": "dark blues, raw guitar, slow tempo",
        "duration": 90
      }
    }
  }'
```

If token mode is enabled, add:

```bash
-H 'Authorization: Bearer <token>'
```

Mark a take as preferred:

```bash
curl -s http://127.0.0.1:38765/mcp \
  -H 'Content-Type: application/json' \
  -d '{
    "jsonrpc": "2.0",
    "id": 6,
    "method": "tools/call",
    "params": {
      "name": "anvil.mark_preferred_take",
      "arguments": {
        "id": "<history-uuid-or-trace-filename>",
        "isPreferred": true,
        "listeningNote": "Best vocal take so far"
      }
    }
  }'
```

Omit `listeningNote` to preserve the existing note. Pass an empty string to
clear it.

Apply one adapter:

```bash
curl -s http://127.0.0.1:38765/mcp \
  -H 'Content-Type: application/json' \
  -d '{
    "jsonrpc": "2.0",
    "id": 7,
    "method": "tools/call",
    "params": {
      "name": "anvil.apply_adapter",
      "arguments": {
        "id": "<adapter-id-or-name>",
        "strength": 0.75
      }
    }
  }'
```

Apply an adapter stack:

```bash
curl -s http://127.0.0.1:38765/mcp \
  -H 'Content-Type: application/json' \
  -d '{
    "jsonrpc": "2.0",
    "id": 8,
    "method": "tools/call",
    "params": {
      "name": "anvil.apply_adapter",
      "arguments": {
        "adapters": [
          { "id": "<primary-adapter-id>", "strength": 0.8 },
          { "id": "<second-adapter-id>", "strength": 0.4 }
        ]
      }
    }
  }'
```

Clear adapters:

```bash
curl -s http://127.0.0.1:38765/mcp \
  -H 'Content-Type: application/json' \
  -d '{
    "jsonrpc": "2.0",
    "id": 9,
    "method": "tools/call",
    "params": {
      "name": "anvil.apply_adapter",
      "arguments": {
        "clear": true
      }
    }
  }'
```

Validate a generation request without starting audio:

```bash
curl -s http://127.0.0.1:38765/mcp \
  -H 'Content-Type: application/json' \
  -d '{
    "jsonrpc": "2.0",
    "id": 10,
    "method": "tools/call",
    "params": {
      "name": "anvil.generate",
      "arguments": {
        "prompt": "slow dark guitar riff",
        "duration": 30,
        "seed": 42,
        "guidancePreset": "constant",
        "dryRun": true
      }
    }
  }'
```

Submit a generation:

```bash
curl -s http://127.0.0.1:38765/mcp \
  -H 'Content-Type: application/json' \
  -d '{
    "jsonrpc": "2.0",
    "id": 11,
    "method": "tools/call",
    "params": {
      "name": "anvil.generate",
      "arguments": {
        "prompt": "slow dark guitar riff",
        "duration": 30,
        "seed": 42
      }
    }
  }'
```

Poll generation status:

```bash
curl -s http://127.0.0.1:38765/mcp \
  -H 'Content-Type: application/json' \
  -d '{
    "jsonrpc": "2.0",
    "id": 12,
    "method": "tools/call",
    "params": {
      "name": "anvil.generation_status",
      "arguments": {}
    }
  }'
```

Release model memory:

```bash
curl -s http://127.0.0.1:38765/mcp \
  -H 'Content-Type: application/json' \
  -d '{
    "jsonrpc": "2.0",
    "id": 13,
    "method": "tools/call",
    "params": {
      "name": "anvil.release_memory",
      "arguments": {
        "confirm": true
      }
    }
  }'
```

## Notes

- The server binds to `127.0.0.1` only.
- It does not expose raw tensors, model weights, or remote/LAN access.
- Destructive and expensive actions are intentionally deferred.
- `anvil.compare_traces` uses the same Model, Adapters, Sampling, Prompt &
  Lyrics, and Audio grouping as the Generation Analyzer compare view.
- `anvil.enhance_prompt` and `anvil.write_lyrics` may load/use the configured
  local LLM or API lyric backends, but their responses are return-only.
- `anvil.mark_preferred_take` is intentionally scoped to trace/history review
  metadata only.
- `anvil.apply_adapter` changes app adapter-selection state but does not start
  generation, load model weights, or alter adapter files.
- `anvil.generate` is non-blocking. It starts the same app generation path and
  returns immediately so MCP clients can poll status or history instead of
  holding a long-running HTTP request open.
- `anvil.release_memory` is intentionally guarded because it cancels active
  generation and unloads resident model state. The next generation or LLM tool
  call reloads what it needs.
