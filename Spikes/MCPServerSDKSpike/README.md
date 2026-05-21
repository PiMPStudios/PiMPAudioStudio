# MCP Server SDK Spike

This isolated Swift 6 package validates whether
`modelcontextprotocol/swift-sdk` can carry Anvil's MCP server surface without
pulling the app target into Swift 6 language mode yet.

## Scope

The spike intentionally exposes only the smallest read-only surface from the
Phase 4 plan:

- tool: `list_history`
- resource list: `anvil://history`
- resource read: `anvil://history`

The package uses a static history provider instead of importing the app target.
That keeps the experiment focused on SDK protocol shape, transport behavior, and
Swift 6 compatibility.

## Findings

- SwiftPM resolved the official SDK at `0.12.1` from the `from: "0.11.0"`
  dependency requirement.
- The SDK provides `Server`, `Tool`, `Resource`, `ListTools`, `CallTool`,
  `ListResources`, and `ReadResource` APIs that map cleanly to Anvil's current
  command-facade shape.
- The SDK supports `structuredContent` for tool results, so Anvil can preserve
  its current text-plus-structured-output behavior.
- The SDK includes `StatelessHTTPServerTransport`, which matches Anvil's current
  request/response-only local endpoint closely: `POST` returns JSON, while `GET`
  and `DELETE` return `405 Method Not Allowed`.
- The SDK also includes `StatefulHTTPServerTransport` for later session, SSE,
  resumability, and server-initiated message support.
- The HTTP transports are framework-agnostic. They process `HTTPRequest` and
  return `HTTPResponse`; they do not bind a port by themselves. Anvil's current
  lightweight socket listener can be adapted to feed requests into the SDK
  transport, or a future NIO listener can be introduced deliberately.
- Token handling can stay outside the SDK transport at first, or move into a
  custom `HTTPRequestValidator` when the bridge is formalized.

## Verification

```bash
swift test --package-path Spikes/MCPServerSDKSpike
swift run --package-path Spikes/MCPServerSDKSpike anvil-mcp-sdk-probe
```

The tests cover:

- SDK server construction with tool/resource capabilities.
- SDK client discovery over `InMemoryTransport`.
- `list_history` tool calls and `anvil://history` resource reads.
- Stateless HTTP transport handling for the same JSON-RPC methods used by
  `scripts/smoke-mcp.sh`: `initialize`, `notifications/initialized`,
  `tools/list`, `tools/call`, `resources/list`, and `resources/read`.

## Recommended Integration Path

This path has been applied in the app target:

1. `AnvilMCPServer` now registers Anvil's existing tools/resources against SDK
   `Server` handlers while keeping `AnvilCommandFacade` unchanged.
2. The old custom JSON-RPC dispatch path now routes through
   `StatelessHTTPServerTransport` behind the existing localhost listener.
3. Legacy `anvil.` and `anvil_` aliases remain inside the SDK `CallTool`
   handler for compatibility.
4. `scripts/smoke-mcp.sh` and a direct SDK-backed curl smoke are the app-level
   verification gates.
5. `StatefulHTTPServerTransport` remains deferred for the richer workstation
   phase that needs SSE, progress, cancellation, and server-initiated
   notifications.
