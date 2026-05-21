import Foundation
import MCP

public enum AnvilSDKServerFactory {
    public static func makeServer(
        historyProvider: any AnvilHistoryProviding,
        version: String = "0.1.0-sdk-spike"
    ) async -> Server {
        let server = Server(
            name: "Anvil SDK Spike",
            version: version,
            instructions: "Prototype Anvil MCP server backed by the official Swift MCP SDK.",
            capabilities: .init(
                resources: .init(subscribe: false, listChanged: false),
                tools: .init(listChanged: false)
            )
        )

        await server.withMethodHandler(ListTools.self) { _ in
            .init(tools: [Self.listHistoryTool])
        }

        await server.withMethodHandler(CallTool.self) { params in
            guard params.name == "list_history" else {
                return .init(
                    content: [
                        .text(
                            text: "Unknown Anvil tool: \(params.name)",
                            annotations: nil,
                            _meta: nil
                        )
                    ],
                    isError: true
                )
            }

            let envelope = try await Self.historyEnvelope(
                historyProvider: historyProvider,
                limit: params.arguments?["limit"]?.intValue ?? 25
            )
            let payload = try Self.encodePayload(envelope)
            return try .init(
                content: [.text(text: payload, annotations: nil, _meta: nil)],
                structuredContent: envelope,
                isError: false
            )
        }

        await server.withMethodHandler(ListResources.self) { _ in
            .init(resources: [Self.historyResource], nextCursor: nil)
        }

        await server.withMethodHandler(ReadResource.self) { params in
            guard params.uri == "anvil://history" else {
                throw MCPError.invalidParams("Unknown Anvil resource URI: \(params.uri)")
            }

            let envelope = try await Self.historyEnvelope(
                historyProvider: historyProvider,
                limit: 200
            )
            let payload = try Self.encodePayload(envelope)
            return .init(contents: [
                Resource.Content.text(payload, uri: params.uri, mimeType: "application/json")
            ])
        }

        return server
    }

    public static let historyResource = Resource(
        name: "history",
        uri: "anvil://history",
        description: "Recent generation rows with model, seed, prompt, audio path, and trace availability.",
        mimeType: "application/json"
    )

    public static let listHistoryTool = Tool(
        name: "list_history",
        description: "Return recent generation history rows with model, seed, prompt summary, " +
            "and trace availability.",
        inputSchema: .object([
            "type": .string("object"),
            "properties": .object([
                "limit": .object([
                    "type": .string("integer"),
                    "minimum": .int(1),
                    "maximum": .int(200),
                    "description": .string("Maximum number of history rows to return. Defaults to 25.")
                ])
            ])
        ]),
        annotations: .init(readOnlyHint: true, destructiveHint: false, idempotentHint: true)
    )

    public static func historyPayload(
        historyProvider: any AnvilHistoryProviding,
        limit: Int
    ) async throws -> String {
        try await encodePayload(
            historyEnvelope(historyProvider: historyProvider, limit: limit)
        )
    }

    public static func historyEnvelope(
        historyProvider: any AnvilHistoryProviding,
        limit: Int
    ) async throws -> AnvilHistoryEnvelope {
        let rows = try await historyProvider.listHistory(limit: limit)
        return AnvilHistoryEnvelope(history: rows)
    }

    private static func encodePayload(_ envelope: AnvilHistoryEnvelope) throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let data = try encoder.encode(envelope)
        return String(decoding: data, as: UTF8.self)
    }
}
