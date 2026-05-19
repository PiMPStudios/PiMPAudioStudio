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

Phase 1 exposes read-only tools:

- `anvil.list_models`
- `anvil.list_adapters`
- `anvil.list_history`
- `anvil.get_trace`
- `anvil.compare_traces`

## Smoke Test With Curl

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

If token mode is enabled, add:

```bash
-H 'Authorization: Bearer <token>'
```

## Notes

- The server binds to `127.0.0.1` only.
- It does not expose raw tensors, model weights, or remote/LAN access.
- Destructive and expensive actions are intentionally deferred.
- `anvil.compare_traces` uses the same Model, Adapters, Sampling, Prompt &
  Lyrics, and Audio grouping as the Generation Analyzer compare view.
