import Foundation
import Darwin
import OSLog

enum AnvilMCPServerState: Equatable {
    case stopped
    case starting
    case running(endpoint: String)
    case failed(String)

    var label: String {
        switch self {
        case .stopped:
            return "Stopped"
        case .starting:
            return "Starting"
        case .running:
            return "Running"
        case .failed:
            return "Failed"
        }
    }
}

@MainActor
final class AnvilMCPServer: ObservableObject {
    private static let enabledKey = "com.pimp.AnvilApp.mcp.enabled"
    private static let tokenRequiredKey = "com.pimp.AnvilApp.mcp.tokenRequired"
    private static let tokenKey = "com.pimp.AnvilApp.mcp.token"
    private static let defaultPort: UInt16 = 38765

    @Published private(set) var isEnabled: Bool
    @Published private(set) var isTokenRequired: Bool
    @Published private(set) var token: String
    @Published private(set) var state: AnvilMCPServerState = .stopped

    private let logger = Logger(subsystem: "com.pimp.AnvilApp", category: "MCP")
    private let facade: AnvilCommandFacade
    private var httpServer: MCPHTTPServer?

    var endpoint: String {
        "http://127.0.0.1:\(Self.defaultPort)/mcp"
    }

    init(facade: AnvilCommandFacade) {
        self.facade = facade
        let defaults = UserDefaults.standard
        self.isEnabled = defaults.bool(forKey: Self.enabledKey)
        self.isTokenRequired = defaults.bool(forKey: Self.tokenRequiredKey)
        if let storedToken = defaults.string(forKey: Self.tokenKey), !storedToken.isEmpty {
            self.token = storedToken
        } else {
            let generated = Self.generateToken()
            self.token = generated
            defaults.set(generated, forKey: Self.tokenKey)
        }

        Task { @MainActor [weak self] in
            guard let self, self.isEnabled else { return }
            self.start()
        }
    }

    deinit {
        httpServer?.stop()
    }

    func setEnabled(_ enabled: Bool) {
        UserDefaults.standard.set(enabled, forKey: Self.enabledKey)
        isEnabled = enabled
        enabled ? start() : stop()
    }

    func setTokenRequired(_ required: Bool) {
        UserDefaults.standard.set(required, forKey: Self.tokenRequiredKey)
        isTokenRequired = required
        if required && token.isEmpty {
            regenerateToken()
        }
    }

    func regenerateToken() {
        let generated = Self.generateToken()
        token = generated
        UserDefaults.standard.set(generated, forKey: Self.tokenKey)
    }

    func stop() {
        httpServer?.stop()
        httpServer = nil
        state = .stopped
    }

    private func start() {
        guard httpServer == nil else {
            state = .running(endpoint: endpoint)
            return
        }

        state = .starting
        do {
            let server = try MCPHTTPServer(
                port: Self.defaultPort,
                tokenProvider: { [weak self] in
                    guard let self, self.isTokenRequired else { return nil }
                    return self.token
                },
                onReady: { [weak self] in
                    self?.state = .running(endpoint: self?.endpoint ?? "")
                },
                onFailed: { [weak self] error in
                    guard let self else { return }
                    self.logger.error("MCP listener failed: \(error.localizedDescription, privacy: .public)")
                    self.httpServer = nil
                    self.state = .failed(error.localizedDescription)
                    self.isEnabled = false
                    UserDefaults.standard.set(false, forKey: Self.enabledKey)
                },
                handler: { [weak self] request in
                    guard let self else {
                        return MCPJSONRPCResponse.error(id: request.id, code: -32603, message: "MCP server is shutting down")
                    }
                    return await self.handle(request)
                }
            )
            try server.start()
            httpServer = server
        } catch {
            logger.error("Failed to start MCP server: \(error.localizedDescription, privacy: .public)")
            state = .failed(error.localizedDescription)
            isEnabled = false
            UserDefaults.standard.set(false, forKey: Self.enabledKey)
        }
    }

    private func handle(_ request: MCPJSONRPCRequest) async -> MCPJSONRPCResponse {
        switch request.method {
        case "initialize":
            return .result(id: request.id, value: [
                "protocolVersion": request.protocolVersion ?? "2025-06-18",
                "capabilities": [
                    "tools": [
                        "listChanged": false
                    ],
                    "resources": [
                        "listChanged": false,
                        "subscribe": false
                    ]
                ],
                "serverInfo": [
                    "name": "Anvil",
                    "version": Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "0.1.0"
                ]
            ])

        case "ping":
            return .result(id: request.id, value: [:])

        case "tools/list":
            return .result(id: request.id, value: ["tools": MCPToolCatalog.tools])

        case "tools/call":
            return await callTool(request)

        case "resources/list":
            return .result(id: request.id, value: [
                "resources": await facade.listResources()
            ])

        case "resources/read":
            return await readResource(request)

        default:
            return .error(id: request.id, code: -32601, message: "Unknown MCP method: \(request.method)")
        }
    }

    private func readResource(_ request: MCPJSONRPCRequest) async -> MCPJSONRPCResponse {
        guard let uri = request.params?["uri"] as? String, !uri.isEmpty else {
            return .error(id: request.id, code: -32602, message: "resources/read requires a resource uri")
        }

        do {
            let payload = try await facade.readResource(uri: uri)
            return .result(id: request.id, value: MCPResourceResult.success(uri: uri, payload: payload))
        } catch {
            return .error(id: request.id, code: -32602, message: error.localizedDescription)
        }
    }

    private func callTool(_ request: MCPJSONRPCRequest) async -> MCPJSONRPCResponse {
        guard let params = request.params,
              let name = params["name"] as? String ?? params["tool_name"] as? String else {
            return .error(id: request.id, code: -32602, message: "tools/call requires a tool name")
        }
        let arguments = params["arguments"] as? [String: Any]
            ?? params["tool_args"] as? [String: Any]
            ?? [:]

        do {
            let structured: Any
            switch name {
            case "anvil.list_models":
                structured = await facade.listModels()
            case "anvil.list_adapters":
                structured = await facade.listAdapters()
            case "anvil.list_history":
                structured = facade.listHistory(limit: intValue(arguments["limit"]) ?? 25)
            case "anvil.get_trace":
                structured = try facade.getTrace(arguments: arguments)
            case "anvil.compare_traces":
                structured = try facade.compareTraces(arguments: arguments)
            case "anvil.mark_preferred_take":
                structured = try facade.markPreferredTake(arguments: arguments)
            case "anvil.enhance_prompt":
                structured = try await facade.enhancePrompt(arguments: arguments)
            case "anvil.write_lyrics":
                structured = try await facade.writeLyrics(arguments: arguments)
            case "anvil.apply_adapter":
                structured = try await facade.applyAdapter(arguments: arguments)
            case "anvil.generate":
                structured = try facade.generate(arguments: arguments)
            case "anvil.generation_status":
                structured = try facade.generationStatus()
            case "anvil.release_memory":
                structured = try await facade.releaseMemory(arguments: arguments)
            default:
                return .error(id: request.id, code: -32601, message: "Unknown Anvil tool: \(name)")
            }

            return .result(id: request.id, value: MCPToolResult.success(structured))
        } catch {
            return .result(id: request.id, value: MCPToolResult.failure(error.localizedDescription))
        }
    }

    private func intValue(_ value: Any?) -> Int? {
        switch value {
        case let value as Int:
            return value
        case let value as NSNumber:
            return value.intValue
        case let value as String:
            return Int(value)
        default:
            return nil
        }
    }

    private static func generateToken() -> String {
        UUID().uuidString.replacingOccurrences(of: "-", with: "").lowercased()
    }
}

private final class MCPHTTPServer {
    private let port: UInt16
    private let tokenProvider: @MainActor () -> String?
    private let onReady: @MainActor () -> Void
    private let onFailed: @MainActor (Error) -> Void
    private let handler: @MainActor (MCPJSONRPCRequest) async -> MCPJSONRPCResponse
    private let acceptQueue = DispatchQueue(label: "com.pimp.AnvilApp.mcp.http.accept")
    private let clientQueue = DispatchQueue(label: "com.pimp.AnvilApp.mcp.http.client", attributes: .concurrent)
    private let stateLock = NSLock()
    private var listenSocket: Int32 = -1
    private var clientSockets = Set<Int32>()
    private var isRunning = false

    init(
        port: UInt16,
        tokenProvider: @escaping @MainActor () -> String?,
        onReady: @escaping @MainActor () -> Void,
        onFailed: @escaping @MainActor (Error) -> Void,
        handler: @escaping @MainActor (MCPJSONRPCRequest) async -> MCPJSONRPCResponse
    ) throws {
        self.port = port
        self.tokenProvider = tokenProvider
        self.onReady = onReady
        self.onFailed = onFailed
        self.handler = handler
    }

    func start() throws {
        let socketFD = Darwin.socket(AF_INET, SOCK_STREAM, 0)
        guard socketFD >= 0 else {
            throw Self.posixError("socket")
        }

        do {
            try configureListenSocket(socketFD)
            try bindLoopback(socketFD)
            guard Darwin.listen(socketFD, SOMAXCONN) == 0 else {
                throw Self.posixError("listen")
            }
        } catch {
            Darwin.close(socketFD)
            throw error
        }

        stateLock.lock()
        listenSocket = socketFD
        isRunning = true
        stateLock.unlock()

        Task { @MainActor in
            self.onReady()
        }

        acceptQueue.async { [weak self] in
            self?.acceptLoop(socketFD)
        }
    }

    func stop() {
        stateLock.lock()
        isRunning = false
        let socketFD = listenSocket
        listenSocket = -1
        let clients = clientSockets
        clientSockets.removeAll()
        stateLock.unlock()

        if socketFD >= 0 {
            Darwin.shutdown(socketFD, SHUT_RDWR)
            Darwin.close(socketFD)
        }
        for client in clients {
            Darwin.shutdown(client, SHUT_RDWR)
            Darwin.close(client)
        }
    }

    private func configureListenSocket(_ socketFD: Int32) throws {
        var reuse: Int32 = 1
        guard setsockopt(socketFD, SOL_SOCKET, SO_REUSEADDR, &reuse, socklen_t(MemoryLayout<Int32>.size)) == 0 else {
            throw Self.posixError("setsockopt SO_REUSEADDR")
        }
    }

    private func bindLoopback(_ socketFD: Int32) throws {
        var address = sockaddr_in()
        address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        address.sin_family = sa_family_t(AF_INET)
        address.sin_port = in_port_t(port).bigEndian
        address.sin_addr = in_addr(s_addr: inet_addr("127.0.0.1"))

        let result = withUnsafePointer(to: &address) { pointer in
            pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) { sockaddrPointer in
                Darwin.bind(socketFD, sockaddrPointer, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        guard result == 0 else {
            throw Self.posixError("bind 127.0.0.1:\(port)")
        }
    }

    private func acceptLoop(_ socketFD: Int32) {
        while isServerRunning(socketFD) {
            var storage = sockaddr_storage()
            var length = socklen_t(MemoryLayout<sockaddr_storage>.size)
            let client = withUnsafeMutablePointer(to: &storage) { pointer in
                pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) { sockaddrPointer in
                    Darwin.accept(socketFD, sockaddrPointer, &length)
                }
            }

            if client < 0 {
                let errnoCode = errno
                if isServerRunning(socketFD) {
                    let error = Self.posixError("accept", errnoCode: errnoCode)
                    Task { @MainActor in
                        self.onFailed(error)
                    }
                }
                return
            }

            configureClientSocket(client)
            addClient(client)
            clientQueue.async { [weak self] in
                self?.receive(on: client, buffer: Data())
            }
        }
    }

    private func configureClientSocket(_ client: Int32) {
        var timeout = timeval(tv_sec: 10, tv_usec: 0)
        setsockopt(client, SOL_SOCKET, SO_RCVTIMEO, &timeout, socklen_t(MemoryLayout<timeval>.size))
        setsockopt(client, SOL_SOCKET, SO_SNDTIMEO, &timeout, socklen_t(MemoryLayout<timeval>.size))

        var noSigpipe: Int32 = 1
        setsockopt(client, SOL_SOCKET, SO_NOSIGPIPE, &noSigpipe, socklen_t(MemoryLayout<Int32>.size))
    }

    private func isServerRunning(_ socketFD: Int32) -> Bool {
        stateLock.lock()
        let running = isRunning && listenSocket == socketFD
        stateLock.unlock()
        return running
    }

    private func addClient(_ client: Int32) {
        stateLock.lock()
        clientSockets.insert(client)
        stateLock.unlock()
    }

    private func closeClient(_ client: Int32) {
        stateLock.lock()
        clientSockets.remove(client)
        stateLock.unlock()
        Darwin.shutdown(client, SHUT_RDWR)
        Darwin.close(client)
    }

    private func receive(on client: Int32, buffer: Data) {
        var buffer = buffer
        var chunk = [UInt8](repeating: 0, count: 64 * 1024)
        let maxRequestBytes = 2 * 1024 * 1024

        while buffer.count < maxRequestBytes {
            let capacity = chunk.count
            let byteCount = chunk.withUnsafeMutableBytes { bytes in
                Darwin.recv(client, bytes.baseAddress, capacity, 0)
            }

            if byteCount > 0 {
                buffer.append(contentsOf: chunk.prefix(byteCount))
                if let request = HTTPRequest(data: buffer) {
                    Task {
                        let response = await self.respond(to: request)
                        self.send(response, on: client)
                        self.closeClient(client)
                    }
                    return
                }
                continue
            }

            if byteCount == 0 {
                closeClient(client)
                return
            }

            if errno == EAGAIN || errno == EWOULDBLOCK {
                send(.badRequest(), on: client)
                closeClient(client)
                return
            }

            closeClient(client)
            return
        }

        send(.badRequest(), on: client)
        closeClient(client)
    }

    private func respond(to request: HTTPRequest) async -> HTTPResponse {
        if request.method == "OPTIONS" {
            return .empty(statusCode: 204, statusText: "No Content")
        }

        if request.method == "GET", request.path == "/health" {
            return .json([
                "status": "ok",
                "endpoint": "/mcp"
            ])
        }

        guard request.method == "POST", request.path == "/mcp" else {
            return .jsonError(statusCode: 404, message: "MCP endpoint is /mcp")
        }

        let expectedToken = await tokenProvider()
        if let expectedToken, !expectedToken.isEmpty {
            let headerToken = request.bearerToken
            guard headerToken == expectedToken else {
                return .jsonError(statusCode: 401, message: "Missing or invalid MCP token")
            }
        }

        guard let rpc = MCPJSONRPCRequest(body: request.body) else {
            return .json(MCPJSONRPCResponse.error(id: nil, code: -32700, message: "Parse error").object)
        }

        if rpc.id == nil {
            return .empty(statusCode: 202, statusText: "Accepted")
        }

        let response = await handler(rpc)
        return .json(response.object)
    }

    private func send(_ response: HTTPResponse, on client: Int32) {
        response.data.withUnsafeBytes { pointer in
            guard var address = pointer.baseAddress, pointer.count > 0 else { return }
            var remaining = pointer.count
            while remaining > 0 {
                let sent = Darwin.send(client, address, remaining, 0)
                guard sent > 0 else { return }
                remaining -= sent
                address += sent
            }
        }
    }

    private static func posixError(_ operation: String, errnoCode: Int32 = errno) -> Error {
        MCPHTTPServerError(operation: operation, errnoCode: errnoCode)
    }
}

private struct MCPHTTPServerError: LocalizedError {
    let operation: String
    let errnoCode: Int32

    var errorDescription: String? {
        "\(operation) failed: \(String(cString: strerror(errnoCode)))"
    }
}

private struct HTTPRequest {
    let method: String
    let path: String
    let headers: [String: String]
    let body: Data

    init?(data: Data) {
        let marker = Data("\r\n\r\n".utf8)
        guard let headerEnd = data.range(of: marker)?.upperBound else { return nil }
        let headerData = data[..<headerEnd]
        guard let headerText = String(data: headerData, encoding: .utf8) else { return nil }

        let lines = headerText.components(separatedBy: "\r\n").filter { !$0.isEmpty }
        guard let requestLine = lines.first else { return nil }
        let parts = requestLine.split(separator: " ", maxSplits: 2).map(String.init)
        guard parts.count >= 2 else { return nil }

        var headers: [String: String] = [:]
        for line in lines.dropFirst() {
            guard let colon = line.firstIndex(of: ":") else { continue }
            let key = line[..<colon].trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            let value = line[line.index(after: colon)...].trimmingCharacters(in: .whitespacesAndNewlines)
            headers[key] = value
        }

        let contentLength = Int(headers["content-length"] ?? "0") ?? 0
        let bodyStart = headerEnd
        guard data.count >= bodyStart + contentLength else { return nil }

        self.method = parts[0]
        self.path = parts[1]
        self.headers = headers
        self.body = Data(data[bodyStart..<(bodyStart + contentLength)])
    }

    var bearerToken: String? {
        guard let authorization = headers["authorization"] else { return nil }
        let prefix = "Bearer "
        guard authorization.hasPrefix(prefix) else { return nil }
        return String(authorization.dropFirst(prefix.count))
    }
}

private struct HTTPResponse {
    let data: Data

    static func json(_ object: Any, statusCode: Int = 200, statusText: String = "OK") -> HTTPResponse {
        let body: Data
        if JSONSerialization.isValidJSONObject(object),
           let encoded = try? JSONSerialization.data(withJSONObject: object, options: [.prettyPrinted, .sortedKeys]) {
            body = encoded
        } else {
            body = Data("{\"error\":\"Invalid JSON response\"}".utf8)
        }
        return response(statusCode: statusCode, statusText: statusText, body: body, contentType: "application/json")
    }

    static func jsonError(statusCode: Int, message: String) -> HTTPResponse {
        json(["error": message], statusCode: statusCode, statusText: message)
    }

    static func badRequest() -> HTTPResponse {
        jsonError(statusCode: 400, message: "Bad Request")
    }

    static func empty(statusCode: Int, statusText: String) -> HTTPResponse {
        response(statusCode: statusCode, statusText: statusText, body: Data(), contentType: "application/json")
    }

    private static func response(statusCode: Int, statusText: String, body: Data, contentType: String) -> HTTPResponse {
        let headers = """
        HTTP/1.1 \(statusCode) \(statusText)\r
        Content-Type: \(contentType)\r
        Content-Length: \(body.count)\r
        Connection: close\r
        Access-Control-Allow-Origin: *\r
        Access-Control-Allow-Headers: Content-Type, Authorization, Mcp-Method\r
        Access-Control-Allow-Methods: POST, GET, OPTIONS\r
        \r

        """
        var data = Data(headers.utf8)
        data.append(body)
        return HTTPResponse(data: data)
    }
}

private struct MCPJSONRPCRequest {
    let id: Any?
    let method: String
    let params: [String: Any]?
    let protocolVersion: String?

    init?(body: Data) {
        guard let object = try? JSONSerialization.jsonObject(with: body) as? [String: Any],
              let method = object["method"] as? String else {
            return nil
        }
        self.id = object["id"]
        self.method = method
        self.params = object["params"] as? [String: Any]
        self.protocolVersion = (object["params"] as? [String: Any])?["protocolVersion"] as? String
    }
}

private struct MCPJSONRPCResponse {
    let object: [String: Any]

    static func result(id: Any?, value: Any) -> MCPJSONRPCResponse {
        MCPJSONRPCResponse(object: [
            "jsonrpc": "2.0",
            "id": id ?? NSNull(),
            "result": value
        ])
    }

    static func error(id: Any?, code: Int, message: String) -> MCPJSONRPCResponse {
        MCPJSONRPCResponse(object: [
            "jsonrpc": "2.0",
            "id": id ?? NSNull(),
            "error": [
                "code": code,
                "message": message
            ]
        ])
    }
}

private enum MCPToolResult {
    static func success(_ structured: Any) -> [String: Any] {
        [
            "content": [
                [
                    "type": "text",
                    "text": prettyJSON(structured)
                ]
            ],
            "structuredContent": structured
        ]
    }

    static func failure(_ message: String) -> [String: Any] {
        [
            "isError": true,
            "content": [
                [
                    "type": "text",
                    "text": message
                ]
            ]
        ]
    }

    private static func prettyJSON(_ object: Any) -> String {
        guard JSONSerialization.isValidJSONObject(object),
              let data = try? JSONSerialization.data(withJSONObject: object, options: [.prettyPrinted, .sortedKeys]),
              let text = String(data: data, encoding: .utf8) else {
            return String(describing: object)
        }
        return text
    }
}

private enum MCPResourceResult {
    static func success(uri: String, payload: Any) -> [String: Any] {
        [
            "contents": [
                [
                    "uri": uri,
                    "mimeType": "application/json",
                    "text": prettyJSON(payload)
                ]
            ]
        ]
    }

    private static func prettyJSON(_ object: Any) -> String {
        guard JSONSerialization.isValidJSONObject(object),
              let data = try? JSONSerialization.data(withJSONObject: object, options: [.prettyPrinted, .sortedKeys]),
              let text = String(data: data, encoding: .utf8) else {
            return String(describing: object)
        }
        return text
    }
}

private enum MCPToolCatalog {
    static let tools: [[String: Any]] = [
        tool(
            name: "anvil.list_models",
            title: "List Models",
            description: "Return Anvil model registry status, active models, readiness, and runtime support notes.",
            properties: [:]
        ),
        tool(
            name: "anvil.list_adapters",
            title: "List Adapters",
            description: "Return installed LoRA adapters with base model, rank, provenance, and checkpoint status.",
            properties: [:]
        ),
        tool(
            name: "anvil.list_history",
            title: "List History",
            description: "Return recent generation history rows with model, seed, prompt summary, and trace availability.",
            properties: [
                "limit": [
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 200,
                    "description": "Maximum number of history rows to return. Defaults to 25."
                ]
            ]
        ),
        tool(
            name: "anvil.get_trace",
            title: "Get Trace",
            description: "Return a generation trace by history id, trace filename, or audio filename.",
            properties: [
                "id": [
                    "type": "string",
                    "description": "History UUID, trace filename, or audio filename."
                ],
                "historyEntryID": ["type": "string"],
                "traceFileName": ["type": "string"],
                "audioFileName": ["type": "string"]
            ]
        ),
        tool(
            name: "anvil.compare_traces",
            title: "Compare Traces",
            description: "Return the same grouped changed-field diff used by the Generation Analyzer compare view.",
            required: ["left", "right"],
            properties: [
                "left": [
                    "type": "string",
                    "description": "Left trace identifier: history UUID, trace filename, or audio filename."
                ],
                "right": [
                    "type": "string",
                    "description": "Right trace identifier: history UUID, trace filename, or audio filename."
                ]
            ]
        ),
        tool(
            name: "anvil.mark_preferred_take",
            title: "Mark Preferred Take",
            description: "Save a preferred flag and optional listening note into an existing generation trace and mirrored history row.",
            properties: [
                "id": [
                    "type": "string",
                    "description": "History UUID, trace filename, or audio filename."
                ],
                "historyEntryID": ["type": "string"],
                "traceFileName": ["type": "string"],
                "audioFileName": ["type": "string"],
                "isPreferred": [
                    "type": "boolean",
                    "description": "Whether the take should be marked preferred. Defaults to true."
                ],
                "preferred": [
                    "type": "boolean",
                    "description": "Alias for isPreferred."
                ],
                "listeningNote": [
                    "type": "string",
                    "description": "Optional listening note. Omit to keep the existing note; pass an empty string to clear it."
                ],
                "note": [
                    "type": "string",
                    "description": "Alias for listeningNote."
                ]
            ]
        ),
        tool(
            name: "anvil.enhance_prompt",
            title: "Enhance Prompt",
            description: "Run the same prompt enhancer used by the Generate view and return the enhanced prompt without mutating app state.",
            required: ["prompt"],
            properties: [
                "prompt": [
                    "type": "string",
                    "description": "Brief prompt or audio description to enhance."
                ],
                "mode": [
                    "type": "string",
                    "enum": ["music", "soundFX", "ambient"],
                    "description": "Generation mode. Defaults to music."
                ],
                "generationMode": [
                    "type": "string",
                    "description": "Alias for mode."
                ]
            ]
        ),
        tool(
            name: "anvil.write_lyrics",
            title: "Write Lyrics",
            description: "Generate cleaned, duration-aware lyrics using the same lyric writer and sanitizer as the Generate view.",
            required: ["prompt", "duration"],
            properties: [
                "prompt": [
                    "type": "string",
                    "description": "Song prompt or creative direction."
                ],
                "style": [
                    "type": "string",
                    "description": "Optional extra style guidance."
                ],
                "duration": [
                    "type": "number",
                    "minimum": 1,
                    "description": "Target song duration in seconds."
                ],
                "durationSeconds": [
                    "type": "number",
                    "minimum": 1,
                    "description": "Alias for duration."
                ]
            ]
        ),
        tool(
            name: "anvil.apply_adapter",
            title: "Apply Adapter",
            description: "Select, stack, or clear ACE-Step LoRA adapters for the next generation. Updates the Generate view when it is open.",
            properties: [
                "id": [
                    "type": "string",
                    "description": "Adapter id or display name to apply as the primary adapter."
                ],
                "adapterID": [
                    "type": "string",
                    "description": "Alias for id."
                ],
                "name": [
                    "type": "string",
                    "description": "Alias for id."
                ],
                "strength": [
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Primary adapter strength. Defaults to 1.0."
                ],
                "adapters": [
                    "type": "array",
                    "description": "Optional stack of adapter objects, each with id/name and optional strength.",
                    "items": [
                        "type": "object",
                        "properties": [
                            "id": ["type": "string"],
                            "adapterID": ["type": "string"],
                            "name": ["type": "string"],
                            "strength": ["type": "number", "minimum": 0, "maximum": 1],
                            "weight": ["type": "number", "minimum": 0, "maximum": 1],
                            "scale": ["type": "number", "minimum": 0, "maximum": 1]
                        ]
                    ]
                ],
                "clear": [
                    "type": "boolean",
                    "description": "Clear all selected adapters when true."
                ]
            ]
        ),
        tool(
            name: "anvil.generate",
            title: "Generate Audio",
            description: "Submit a generation request through the same path as the Generate view. Returns immediately; poll anvil.generation_status or history for completion.",
            required: ["prompt"],
            properties: [
                "prompt": [
                    "type": "string",
                    "description": "Prompt or audio description."
                ],
                "text": [
                    "type": "string",
                    "description": "Alias for prompt."
                ],
                "lyrics": [
                    "type": "string",
                    "description": "Optional lyrics for music generation."
                ],
                "duration": [
                    "type": "number",
                    "minimum": 1,
                    "description": "Duration in seconds. Defaults to 30. Stable Audio requests are capped at 47 seconds."
                ],
                "seed": [
                    "description": "Integer seed, 'random', empty, or omitted for random."
                ],
                "mode": [
                    "type": "string",
                    "enum": ["music", "soundFX", "ambient"],
                    "description": "Resolved generation mode. Defaults to music."
                ],
                "pipeline": [
                    "type": "string",
                    "enum": ["auto", "aceStep", "stableAudio"],
                    "description": "Pipeline override. Defaults to auto."
                ],
                "guidanceScale": [
                    "type": "number",
                    "description": "CFG guidance scale. Defaults to 7.5."
                ],
                "guidancePreset": [
                    "type": "string",
                    "enum": ["constant", "lateEmphasis", "midToLate", "softLate"],
                    "description": "Guidance schedule preset. Defaults to softLate."
                ],
                "method": [
                    "type": "string",
                    "enum": ["ode", "sde"],
                    "description": "Diffusion method. Defaults to ode."
                ],
                "precision": [
                    "type": "string",
                    "enum": ["float32", "float16"],
                    "description": "Runtime precision. Defaults to float32."
                ],
                "steps": [
                    "description": "Positive integer step override, 'auto', or omitted for auto."
                ],
                "shift": [
                    "type": "number",
                    "description": "ACE-Step shift override. Defaults to 3.0."
                ],
                "referenceAudioPath": [
                    "type": "string",
                    "description": "Optional local audio path for cover-song generation."
                ],
                "coverStrength": [
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Cover-song blend strength. Defaults to 0.7."
                ],
                "dryRun": [
                    "type": "boolean",
                    "description": "Validate and echo the resolved request without starting generation."
                ]
            ]
        ),
        tool(
            name: "anvil.generation_status",
            title: "Generation Status",
            description: "Return the current generation state and latest completed result metadata.",
            properties: [:]
        ),
        tool(
            name: "anvil.release_memory",
            title: "Release Memory",
            description: "Cancel active generation if needed, unload model weights, and flush MLX GPU cache. Requires confirm: true.",
            required: ["confirm"],
            properties: [
                "confirm": [
                    "type": "boolean",
                    "description": "Must be true to release memory."
                ]
            ]
        )
    ]

    private static func tool(
        name: String,
        title: String,
        description: String,
        required: [String] = [],
        properties: [String: Any]
    ) -> [String: Any] {
        [
            "name": name,
            "title": title,
            "description": description,
            "inputSchema": [
                "type": "object",
                "additionalProperties": false,
                "required": required,
                "properties": properties
            ]
        ]
    }
}
