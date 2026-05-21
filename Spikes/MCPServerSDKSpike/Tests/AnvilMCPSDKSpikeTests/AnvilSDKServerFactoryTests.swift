import Foundation
import MCP
import Testing
@testable import AnvilMCPSDKSpike

@Suite("Anvil MCP SDK Spike")
struct AnvilSDKServerFactoryTests {
    @Test("history payload matches Anvil MCP envelope")
    func historyPayloadMatchesEnvelope() async throws {
        let payload = try await AnvilSDKServerFactory.historyPayload(
            historyProvider: StaticAnvilHistoryProvider(),
            limit: 25
        )

        let envelope = try JSONDecoder().decode(
            AnvilHistoryEnvelope.self,
            from: Data(payload.utf8)
        )

        #expect(envelope.count == 1)
        #expect(envelope.history.first?.fileName == "2026-05-20_sparse-dark-guitar.wav")
        #expect(envelope.history.first?.modelID == "ace-step-v1.5")
    }

    @Test("SDK catalog exposes the intended tiny read-only surface")
    func sdkCatalogMatchesSpikeScope() async {
        let tool = AnvilSDKServerFactory.listHistoryTool
        let resource = AnvilSDKServerFactory.historyResource

        #expect(tool.name == "list_history")
        #expect(resource.uri == "anvil://history")
        #expect(resource.mimeType == "application/json")
    }

    @Test("SDK server can be constructed with tool and resource capabilities")
    func sdkServerConstruction() async {
        let server = await AnvilSDKServerFactory.makeServer(
            historyProvider: StaticAnvilHistoryProvider()
        )

        _ = server
    }

    @Test("SDK client discovers and calls the tiny read-only surface")
    func sdkClientDiscovery() async throws {
        let transports = await InMemoryTransport.createConnectedPair()
        let server = await AnvilSDKServerFactory.makeServer(
            historyProvider: StaticAnvilHistoryProvider()
        )
        let client = Client(name: "AnvilSDKSpikeTests", version: "0.1.0")

        try await server.start(transport: transports.server)
        do {
            try await client.connect(transport: transports.client)

            let tools = try await client.listTools().tools
            #expect(tools.map(\.name) == ["list_history"])

            let resources = try await client.listResources().resources
            #expect(resources.map(\.uri) == ["anvil://history"])

            let toolResult = try await client.callTool(
                name: "list_history",
                arguments: ["limit": 1]
            )
            #expect(toolResult.isError == false)
            #expect(toolResult.content.first?.textValue?.contains("ace-step-v1.5") == true)

            let resourceContents = try await client.readResource(uri: "anvil://history")
            #expect(resourceContents.first?.text?.contains("2026-05-20_sparse-dark-guitar.wav") == true)

            await client.disconnect()
            await server.stop()
        } catch {
            await client.disconnect()
            await server.stop()
            throw error
        }
    }

    @Test("SDK stateless HTTP transport matches smoke-script method shape")
    func statelessHTTPTransportMatchesSmokeShape() async throws {
        let transport = StatelessHTTPServerTransport()
        let server = await AnvilSDKServerFactory.makeServer(
            historyProvider: StaticAnvilHistoryProvider()
        )

        try await server.start(transport: transport)
        do {
            let initialize = await transport.handleRequest(try rpcRequest(
                id: 1,
                method: "initialize",
                params: [
                    "protocolVersion": "2025-06-18",
                    "capabilities": [:],
                    "clientInfo": [
                        "name": "AnvilSDKSpikeHTTPTests",
                        "version": "0.1.0",
                    ],
                ]
            ))
            #expect(initialize.statusCode == 200)

            let initialized = await transport.handleRequest(try rpcNotification(
                method: "notifications/initialized"
            ))
            #expect(initialized.statusCode == 202)

            let tools = try jsonObject(await transport.handleRequest(try rpcRequest(
                id: 2,
                method: "tools/list"
            )))
            let toolResult = try #require(tools["result"] as? [String: Any])
            let toolList = try #require(toolResult["tools"] as? [[String: Any]])
            #expect(toolList.first?["name"] as? String == "list_history")

            let resources = try jsonObject(await transport.handleRequest(try rpcRequest(
                id: 3,
                method: "resources/list"
            )))
            let resourceResult = try #require(resources["result"] as? [String: Any])
            let resourceList = try #require(resourceResult["resources"] as? [[String: Any]])
            #expect(resourceList.first?["uri"] as? String == "anvil://history")

            let call = try jsonObject(await transport.handleRequest(try rpcRequest(
                id: 4,
                method: "tools/call",
                params: [
                    "name": "list_history",
                    "arguments": ["limit": 1],
                ]
            )))
            let callResult = try #require(call["result"] as? [String: Any])
            let structuredContent = try #require(callResult["structuredContent"] as? [String: Any])
            #expect(structuredContent["count"] as? Int == 1)

            let read = try jsonObject(await transport.handleRequest(try rpcRequest(
                id: 5,
                method: "resources/read",
                params: ["uri": "anvil://history"]
            )))
            let readResult = try #require(read["result"] as? [String: Any])
            let contents = try #require(readResult["contents"] as? [[String: Any]])
            #expect(contents.first?["mimeType"] as? String == "application/json")
            #expect((contents.first?["text"] as? String)?.contains("ace-step-v1.5") == true)

            await server.stop()
        } catch {
            await server.stop()
            throw error
        }
    }
}

private extension Tool.Content {
    var textValue: String? {
        if case .text(let text, _, _) = self {
            return text
        }
        return nil
    }
}

private func rpcRequest(
    id: Int,
    method: String,
    params: [String: Any]? = nil
) throws -> HTTPRequest {
    try rpcHTTPRequest(id: id, method: method, params: params)
}

private func rpcNotification(
    method: String,
    params: [String: Any]? = nil
) throws -> HTTPRequest {
    try rpcHTTPRequest(id: nil, method: method, params: params)
}

private func rpcHTTPRequest(
    id: Int?,
    method: String,
    params: [String: Any]?
) throws -> HTTPRequest {
    var object: [String: Any] = [
        "jsonrpc": "2.0",
        "method": method,
    ]
    if let id {
        object["id"] = id
    }
    if let params {
        object["params"] = params
    }

    return HTTPRequest(
        method: "POST",
        headers: [
            HTTPHeaderName.accept: "application/json",
            HTTPHeaderName.contentType: "application/json",
            HTTPHeaderName.protocolVersion: "2025-06-18",
            HTTPHeaderName.host: "127.0.0.1:38765",
            HTTPHeaderName.origin: "http://127.0.0.1:38765",
        ],
        body: try JSONSerialization.data(withJSONObject: object),
        path: "/mcp"
    )
}

private func jsonObject(_ response: HTTPResponse) throws -> [String: Any] {
    #expect(response.statusCode == 200)
    let data = try #require(response.bodyData)
    let object = try JSONSerialization.jsonObject(with: data)
    return try #require(object as? [String: Any])
}
