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

`anvil.enhance_prompt` and `anvil.write_lyrics` call the same local/API
intelligence services used by the Generate view. They return text only and do
not update the current prompt, lyric field, history, traces, models, or audio.

It also exposes one controlled metadata write:

- `anvil.mark_preferred_take`

`anvil.mark_preferred_take` only updates review metadata on an existing trace:
the preferred flag and optional listening note. It mirrors the same values into
`history.json`, matching the Generation Analyzer's `Mark Preferred` / notes
behavior. It does not modify audio files, models, adapters, tensors, or
generation settings.

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
