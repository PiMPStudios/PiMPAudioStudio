import Foundation
import Darwin
import MCP
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
    private var sdkBridge: AnvilMCPSDKBridge?
    private var startupTask: Task<Void, Never>?

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
        startupTask?.cancel()
        httpServer?.stop()
        if let sdkBridge {
            Task { await sdkBridge.stop() }
        }
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
        startupTask?.cancel()
        startupTask = nil
        httpServer?.stop()
        httpServer = nil
        if let sdkBridge {
            Task { await sdkBridge.stop() }
        }
        sdkBridge = nil
        state = .stopped
    }

    private func start() {
        guard httpServer == nil, startupTask == nil else {
            state = .running(endpoint: endpoint)
            return
        }

        state = .starting
        startupTask = Task { @MainActor [weak self] in
            guard let self else { return }

            let bridge = AnvilMCPSDKBridge(
                facade: self.facade,
                version: Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "0.1.0"
            )
            var startedServer: MCPHTTPServer?

            do {
                try Task.checkCancellation()
                try await bridge.start()
                try Task.checkCancellation()

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
                        self?.failStartup(error)
                    },
                    handler: { sdkRequest in
                        await bridge.handle(sdkRequest)
                    }
                )
                try server.start()
                startedServer = server
                try Task.checkCancellation()

                self.sdkBridge = bridge
                self.httpServer = server
                self.startupTask = nil
            } catch is CancellationError {
                startedServer?.stop()
                await bridge.stop()
                self.startupTask = nil
            } catch {
                startedServer?.stop()
                await bridge.stop()
                self.startupTask = nil
                self.failStartup(error)
            }
        }
    }

    private func failStartup(_ error: Error) {
        logger.error("MCP server failed: \(error.localizedDescription, privacy: .public)")
        httpServer?.stop()
        httpServer = nil
        if let sdkBridge {
            Task { await sdkBridge.stop() }
        }
        sdkBridge = nil
        state = .failed(error.localizedDescription)
        isEnabled = false
        UserDefaults.standard.set(false, forKey: Self.enabledKey)
    }

    private static func generateToken() -> String {
        UUID().uuidString.replacingOccurrences(of: "-", with: "").lowercased()
    }
}

@MainActor
private final class AnvilMCPSDKBridge {
    private let facade: AnvilCommandFacade
    private let version: String
    private var server: Server
    private var transport: StatefulHTTPServerTransport
    private var isStarted = false
    private var hasActiveSession = false

    init(facade: AnvilCommandFacade, version: String) {
        self.facade = facade
        self.version = version
        let components = Self.makeServer(version: version)
        self.server = components.server
        self.transport = components.transport
    }

    private static func makeServer(version: String) -> (server: Server, transport: StatefulHTTPServerTransport) {
        let server = Server(
            name: "Anvil",
            version: version,
            capabilities: .init(
                resources: .init(subscribe: false, listChanged: false),
                tools: .init(listChanged: false)
            )
        )
        return (server, StatefulHTTPServerTransport())
    }

    func start() async throws {
        guard !isStarted else { return }
        await registerHandlers()
        try await server.start(transport: transport)
        isStarted = true
    }

    func stop() async {
        guard isStarted else { return }
        await server.stop()
        isStarted = false
        hasActiveSession = false
    }

    func handle(_ request: MCP.HTTPRequest) async -> MCP.HTTPResponse {
        let response = await transport.handleRequest(request)

        if isInitializeRequest(request), response.statusCode == 200 {
            hasActiveSession = true
        } else if request.method.uppercased() == "DELETE", (200..<300).contains(response.statusCode) {
            hasActiveSession = false
            _ = await resetSessionTransport()
        }

        return response
    }

    private func resetSessionTransport() async -> MCP.HTTPResponse? {
        if isStarted {
            await server.stop()
            isStarted = false
        }

        let components = Self.makeServer(version: version)
        server = components.server
        transport = components.transport
        hasActiveSession = false

        do {
            await registerHandlers()
            try await server.start(transport: transport)
            isStarted = true
            return nil
        } catch {
            return .error(statusCode: 500, .internalError("Failed to reset MCP session transport: \(error.localizedDescription)"))
        }
    }

    private func isInitializeRequest(_ request: MCP.HTTPRequest) -> Bool {
        guard request.method.uppercased() == "POST",
              let body = request.body,
              let json = try? JSONSerialization.jsonObject(with: body) as? [String: Any],
              let method = json["method"] as? String else {
            return false
        }
        return method == "initialize"
    }

    private func registerHandlers() async {
        await server.withMethodHandler(ListTools.self) { _ in
            .init(tools: MCPToolCatalog.sdkTools)
        }

        await server.withMethodHandler(CallTool.self) { [weak self] params in
            guard let self else {
                return .init(
                    content: [.text(text: "MCP server is shutting down", annotations: nil, _meta: nil)],
                    isError: true
                )
            }
            return await self.callTool(params)
        }

        await server.withMethodHandler(ListResources.self) { [weak self] _ in
            guard let self else {
                return .init(resources: [], nextCursor: nil)
            }
            let resources = await self.facade.listResources()
            return .init(resources: resources.compactMap(MCPResourceCatalog.sdkResource), nextCursor: nil)
        }

        await server.withMethodHandler(ReadResource.self) { [weak self] params in
            guard let self else {
                throw MCPError.internalError("MCP server is shutting down")
            }
            return try await self.readResource(params)
        }
    }

    private func callTool(_ params: CallTool.Parameters) async -> CallTool.Result {
        let arguments = MCPValueConverter.dictionary(from: params.arguments)
        let toolName = canonicalToolName(params.name)
        let progressToken = params._meta?.progressToken

        await sendProgress(
            progressToken,
            progress: 0,
            total: 1,
            message: "Starting \(toolName)"
        )

        do {
            let structured = try await executeTool(
                name: toolName,
                arguments: arguments
            )
            let structuredContent: Value? = try MCPValueConverter.value(from: structured)
            if let operationID = backgroundOperationID(from: structured) {
                await sendProgress(
                    progressToken,
                    progress: 1,
                    total: 1,
                    message: "Submitted \(toolName) as \(operationID); poll the operation for background progress."
                )
            } else {
                await sendProgress(
                    progressToken,
                    progress: 1,
                    total: 1,
                    message: "Completed \(toolName)"
                )
            }
            return .init(
                content: [.text(text: MCPJSON.prettyJSON(structured), annotations: nil, _meta: nil)],
                structuredContent: structuredContent,
                isError: false
            )
        } catch {
            await sendProgress(
                progressToken,
                progress: 1,
                total: 1,
                message: "Failed \(toolName): \(error.localizedDescription)"
            )
            return .init(
                content: [.text(text: error.localizedDescription, annotations: nil, _meta: nil)],
                isError: true
            )
        }
    }

    private func sendProgress(
        _ progressToken: ProgressToken?,
        progress: Double,
        total: Double?,
        message: String
    ) async {
        guard let progressToken else { return }
        let notification = Message<ProgressNotification>(
            method: ProgressNotification.name,
            params: .init(
                progressToken: progressToken,
                progress: progress,
                total: total,
                message: message
            )
        )
        try? await server.notify(notification)
    }

    private func backgroundOperationID(from structured: Any) -> String? {
        guard let payload = structured as? [String: Any] else { return nil }
        if let operationID = nonEmptyString(payload["operationID"]) {
            return operationID
        }
        if let generation = payload["generation"] as? [String: Any],
           let operationID = nonEmptyString(generation["operationID"]) {
            return operationID
        }
        return nil
    }

    private func nonEmptyString(_ value: Any?) -> String? {
        let string = stringValue(value)?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        return string.isEmpty ? nil : string
    }

    private func readResource(_ params: ReadResource.Parameters) async throws -> ReadResource.Result {
        do {
            let payload = try await facade.readResource(uri: params.uri)
            return .init(contents: [
                .text(
                    MCPJSON.prettyJSON(payload),
                    uri: params.uri,
                    mimeType: "application/json"
                )
            ])
        } catch {
            throw MCPError.invalidParams(error.localizedDescription)
        }
    }

    private func executeTool(name: String, arguments: [String: Any]) async throws -> Any {
        switch name {
        case "list_models":
            return await facade.listModels()
        case "list_adapters":
            return await facade.listAdapters()
        case "list_history":
            return facade.listHistory(limit: intValue(arguments["limit"]) ?? 25)
        case "get_trace":
            return try facade.getTrace(arguments: arguments)
        case "compare_traces":
            return try facade.compareTraces(arguments: arguments)
        case "mark_preferred_take":
            return try facade.markPreferredTake(arguments: arguments)
        case "enhance_prompt":
            return try await facade.enhancePrompt(arguments: arguments)
        case "write_lyrics":
            return try await facade.writeLyrics(arguments: arguments)
        case "apply_adapter":
            return try await facade.applyAdapter(arguments: arguments)
        case "generate_song":
            return try await facade.generateSong(arguments: arguments)
        case "generate":
            return try facade.generate(arguments: arguments)
        case "generation_status":
            return try facade.generationStatus()
        case "editor_status":
            return try await facade.editorStatus()
        case "load_editor_audio":
            return try facade.loadEditorAudio(arguments: arguments)
        case "submit_editor_inpaint":
            return try await facade.submitEditorInpaint(arguments: arguments)
        case "render_editor_audio":
            return try await facade.renderEditorAudio(arguments: arguments)
        case "list_operations":
            return facade.listOperations(arguments: arguments)
        case "get_operation":
            return try facade.getOperation(arguments: arguments)
        case "cancel_operation":
            return try facade.cancelOperation(arguments: arguments)
        case "release_memory":
            return try await facade.releaseMemory(arguments: arguments)
        default:
            throw MCPError.methodNotFound("Unknown Anvil tool: \(name)")
        }
    }

    private func canonicalToolName(_ name: String) -> String {
        if name.hasPrefix("anvil.") {
            return String(name.dropFirst("anvil.".count))
        }
        if name.hasPrefix("anvil_") {
            return String(name.dropFirst("anvil_".count))
        }
        return name
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

    private func stringValue(_ value: Any?) -> String? {
        switch value {
        case _ as NSNull:
            return nil
        case let value as String:
            return value
        case let value as CustomStringConvertible:
            return value.description
        default:
            return nil
        }
    }
}

private enum MCPValueConverter {
    static func dictionary(from values: [String: Value]?) -> [String: Any] {
        values?.mapValues(anyValue) ?? [:]
    }

    static func value(from any: Any) throws -> Value {
        switch any {
        case _ as NSNull:
            return .null
        case let value as Bool:
            return .bool(value)
        case let value as Int:
            return .int(value)
        case let value as Int64:
            return .int(Int(value))
        case let value as Double:
            return .double(value)
        case let value as Float:
            return .double(Double(value))
        case let value as String:
            return .string(value)
        case let value as NSNumber:
            return numberValue(value)
        case let value as [Any]:
            return .array(try value.map(Self.value))
        case let value as [String: Any]:
            return .object(try value.mapValues(Self.value))
        default:
            return .string(String(describing: any))
        }
    }

    private static func anyValue(_ value: Value) -> Any {
        switch value {
        case .null:
            return NSNull()
        case .bool(let bool):
            return bool
        case .int(let int):
            return int
        case .double(let double):
            return double
        case .string(let string):
            return string
        case .data(_, let data):
            return data.base64EncodedString()
        case .array(let array):
            return array.map(anyValue)
        case .object(let object):
            return object.mapValues(anyValue)
        }
    }

    private static func numberValue(_ value: NSNumber) -> Value {
        if CFGetTypeID(value) == CFBooleanGetTypeID() {
            return .bool(value.boolValue)
        }

        let double = value.doubleValue
        if double.rounded() == double {
            return .int(value.intValue)
        }
        return .double(double)
    }
}

private enum MCPJSON {
    static func prettyJSON(_ object: Any) -> String {
        guard JSONSerialization.isValidJSONObject(object),
              let data = try? JSONSerialization.data(
                withJSONObject: object,
                options: [.prettyPrinted, .sortedKeys]
              ),
              let text = String(data: data, encoding: .utf8) else {
            return String(describing: object)
        }
        return text
    }
}

private enum MCPResourceCatalog {
    static func sdkResource(_ definition: [String: Any]) -> Resource? {
        guard let name = definition["name"] as? String,
              let uri = definition["uri"] as? String else {
            return nil
        }

        return Resource(
            name: name,
            uri: uri,
            title: definition["title"] as? String,
            description: definition["description"] as? String,
            mimeType: definition["mimeType"] as? String ?? "application/json"
        )
    }
}

private final class MCPHTTPServer {
    private let port: UInt16
    private let tokenProvider: @MainActor () -> String?
    private let onReady: @MainActor () -> Void
    private let onFailed: @MainActor (Error) -> Void
    private let handler: @MainActor (MCP.HTTPRequest) async -> MCP.HTTPResponse
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
        handler: @escaping @MainActor (MCP.HTTPRequest) async -> MCP.HTTPResponse
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
                if let request = RawHTTPRequest(data: buffer) {
                    Task {
                        let response = await self.respond(to: request)
                        await self.send(response, on: client)
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
                sendAndClose(.badRequest(), on: client)
                return
            }

            closeClient(client)
            return
        }

        sendAndClose(.badRequest(), on: client)
    }

    private func respond(to request: RawHTTPRequest) async -> RawHTTPResponse {
        guard request.hasTrustedOrigin else {
            return .jsonError(statusCode: 403, message: "Forbidden Origin")
        }

        if request.method == "OPTIONS" {
            return .empty(statusCode: 204, statusText: "No Content")
        }

        if request.method == "GET", request.path == "/health" {
            return .json([
                "status": "ok",
                "endpoint": "/mcp"
            ])
        }

        guard request.path == "/mcp" else {
            return .jsonError(statusCode: 404, message: "MCP endpoint is /mcp")
        }

        let expectedToken = await tokenProvider()
        if let expectedToken, !expectedToken.isEmpty {
            let headerToken = request.bearerToken
            guard headerToken == expectedToken else {
                return .jsonError(statusCode: 401, message: "Missing or invalid MCP token")
            }
        }

        let sdkResponse = await handler(request.sdkRequest)
        return .sdk(sdkResponse)
    }

    private func send(_ response: RawHTTPResponse, on client: Int32) async {
        switch response {
        case .fixed(let data):
            _ = sendData(data, on: client)
        case .stream(let header, let stream):
            guard sendData(header, on: client) else { return }
            do {
                for try await chunk in stream {
                    guard sendData(chunk, on: client) else { return }
                }
            } catch {
                return
            }
        }
    }

    private func sendAndClose(_ response: RawHTTPResponse, on client: Int32) {
        Task {
            await self.send(response, on: client)
            self.closeClient(client)
        }
    }

    private func sendData(_ data: Data, on client: Int32) -> Bool {
        if data.isEmpty { return true }
        return data.withUnsafeBytes { pointer in
            guard var address = pointer.baseAddress, pointer.count > 0 else { return true }
            var remaining = pointer.count
            while remaining > 0 {
                let sent = Darwin.send(client, address, remaining, 0)
                guard sent > 0 else { return false }
                remaining -= sent
                address += sent
            }
            return true
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

private struct RawHTTPRequest {
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
        self.path = parts[1].split(separator: "?", maxSplits: 1).first.map(String.init) ?? parts[1]
        self.headers = headers
        self.body = Data(data[bodyStart..<(bodyStart + contentLength)])
    }

    var bearerToken: String? {
        guard let authorization = headers["authorization"] else { return nil }
        let prefix = "Bearer "
        guard authorization.hasPrefix(prefix) else { return nil }
        return String(authorization.dropFirst(prefix.count))
    }

    var hasTrustedOrigin: Bool {
        guard let origin = headers["origin"]?.trimmingCharacters(in: .whitespacesAndNewlines),
              !origin.isEmpty else {
            return true
        }
        guard let url = URL(string: origin),
              let host = url.host?.lowercased() else {
            return false
        }
        return host == "localhost" || host == "127.0.0.1" || host == "::1"
    }

    var sdkRequest: MCP.HTTPRequest {
        MCP.HTTPRequest(
            method: method,
            headers: headers,
            body: body,
            path: path
        )
    }
}

private enum RawHTTPResponse {
    case fixed(Data)
    case stream(header: Data, body: AsyncThrowingStream<Data, Error>)

    static func json(_ object: Any, statusCode: Int = 200, statusText: String = "OK") -> RawHTTPResponse {
        let body: Data
        if JSONSerialization.isValidJSONObject(object),
           let encoded = try? JSONSerialization.data(withJSONObject: object, options: [.prettyPrinted, .sortedKeys]) {
            body = encoded
        } else {
            body = Data("{\"error\":\"Invalid JSON response\"}".utf8)
        }
        return response(statusCode: statusCode, statusText: statusText, body: body, contentType: "application/json")
    }

    static func jsonError(statusCode: Int, message: String) -> RawHTTPResponse {
        json(["error": message], statusCode: statusCode, statusText: message)
    }

    static func badRequest() -> RawHTTPResponse {
        jsonError(statusCode: 400, message: "Bad Request")
    }

    static func empty(statusCode: Int, statusText: String) -> RawHTTPResponse {
        response(statusCode: statusCode, statusText: statusText, body: Data(), contentType: "application/json")
    }

    static func sdk(_ response: MCP.HTTPResponse) -> RawHTTPResponse {
        if case .stream(let stream, let headers) = response {
            let contentType = headers[MCP.HTTPHeaderName.contentType] ?? "text/event-stream"
            return self.stream(
                statusCode: response.statusCode,
                statusText: statusText(for: response.statusCode),
                contentType: contentType,
                body: stream,
                extraHeaders: headers
            )
        }

        let body = response.bodyData ?? Data()
        let contentType = response.headers[MCP.HTTPHeaderName.contentType] ?? "application/json"
        return self.response(
            statusCode: response.statusCode,
            statusText: statusText(for: response.statusCode),
            body: body,
            contentType: contentType,
            extraHeaders: response.headers
        )
    }

    private static func stream(
        statusCode: Int,
        statusText: String,
        contentType: String,
        body: AsyncThrowingStream<Data, Error>,
        extraHeaders: [String: String] = [:]
    ) -> RawHTTPResponse {
        let headers = responseHeaders(
            contentType: contentType,
            contentLength: nil,
            connection: "keep-alive",
            extraHeaders: extraHeaders
        )
        return .stream(header: headerData(statusCode: statusCode, statusText: statusText, headers: headers), body: body)
    }

    private static func response(
        statusCode: Int,
        statusText: String,
        body: Data,
        contentType: String,
        extraHeaders: [String: String] = [:]
    ) -> RawHTTPResponse {
        let headers = responseHeaders(
            contentType: contentType,
            contentLength: "\(body.count)",
            connection: "close",
            extraHeaders: extraHeaders
        )
        var data = headerData(statusCode: statusCode, statusText: statusText, headers: headers)
        data.append(body)
        return .fixed(data)
    }

    private static func responseHeaders(
        contentType: String,
        contentLength: String?,
        connection: String,
        extraHeaders: [String: String]
    ) -> [String: String] {
        var headers = [
            "Content-Type": contentType,
            "Connection": connection,
            "MCP-Protocol-Version": "2025-06-18",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Accept, Content-Type, Authorization, Mcp-Method, MCP-Protocol-Version, MCP-Session-Id, Last-Event-ID",
            "Access-Control-Allow-Methods": "POST, GET, DELETE, OPTIONS",
            "Allow": "POST, GET, DELETE, OPTIONS"
        ]
        if let contentLength {
            headers["Content-Length"] = contentLength
        }
        headers.merge(extraHeaders) { _, sdkValue in sdkValue }
        if contentLength == nil {
            headers.removeValue(forKey: "Content-Length")
        }
        return headers
    }

    private static func headerData(statusCode: Int, statusText: String, headers: [String: String]) -> Data {
        let headerLines = headers
            .sorted { $0.key < $1.key }
            .map { "\($0.key): \($0.value)" }
            .joined(separator: "\r\n")
        return Data("HTTP/1.1 \(statusCode) \(statusText)\r\n\(headerLines)\r\n\r\n".utf8)
    }

    private static func statusText(for statusCode: Int) -> String {
        switch statusCode {
        case 200: "OK"
        case 202: "Accepted"
        case 204: "No Content"
        case 400: "Bad Request"
        case 401: "Unauthorized"
        case 403: "Forbidden"
        case 404: "Not Found"
        case 405: "Method Not Allowed"
        case 406: "Not Acceptable"
        case 415: "Unsupported Media Type"
        case 421: "Misdirected Request"
        case 500: "Internal Server Error"
        default: "HTTP \(statusCode)"
        }
    }
}

private enum MCPToolCatalog {
    static var sdkTools: [Tool] {
        tools.compactMap(sdkTool)
    }

    static let tools: [[String: Any]] = [
        tool(
            name: "list_models",
            title: "List Models",
            description: "Return Anvil model registry status, active models, readiness, and runtime support notes.",
            properties: [:]
        ),
        tool(
            name: "list_adapters",
            title: "List Adapters",
            description: "Return installed LoRA adapters with base model, rank, provenance, and checkpoint status.",
            properties: [:]
        ),
        tool(
            name: "list_history",
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
            name: "get_trace",
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
            name: "compare_traces",
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
            name: "mark_preferred_take",
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
            name: "enhance_prompt",
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
            name: "write_lyrics",
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
            name: "apply_adapter",
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
            name: "generate_song",
            title: "Generate Song",
            description: "Orchestrate a song generation: optionally enhance the prompt, optionally write duration-aware lyrics, optionally apply adapters, then submit generation.",
            required: ["prompt"],
            properties: [
                "prompt": [
                    "type": "string",
                    "description": "Prompt or creative direction."
                ],
                "text": [
                    "type": "string",
                    "description": "Alias for prompt."
                ],
                "enhancePrompt": [
                    "type": "boolean",
                    "description": "Run the prompt enhancer before generation. Defaults to false."
                ],
                "enhance": [
                    "type": "boolean",
                    "description": "Alias for enhancePrompt."
                ],
                "writeLyrics": [
                    "type": "boolean",
                    "description": "Write duration-aware lyrics when no lyrics are supplied. Defaults to false."
                ],
                "generateLyrics": [
                    "type": "boolean",
                    "description": "Alias for writeLyrics."
                ],
                "lyrics": [
                    "type": "string",
                    "description": "Optional lyrics. Supplied lyrics are used as-is and prevent automatic lyric writing."
                ],
                "style": [
                    "type": "string",
                    "description": "Optional lyric style guidance."
                ],
                "lyricStyle": [
                    "type": "string",
                    "description": "Alias for style."
                ],
                "adapter": [
                    "type": "string",
                    "description": "Optional adapter id or display name to apply before generation."
                ],
                "lora": [
                    "type": "string",
                    "description": "Alias for adapter."
                ],
                "adapterID": [
                    "type": "string",
                    "description": "Alias for adapter."
                ],
                "strength": [
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Adapter strength when adapter/lora is supplied."
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
                "clearAdapter": [
                    "type": "boolean",
                    "description": "Clear selected adapters before generation."
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
                "dryRun": [
                    "type": "boolean",
                    "description": "Resolve orchestration and validate the generation request without starting generation or applying adapters."
                ]
            ]
        ),
        tool(
            name: "generate",
            title: "Generate Audio",
            description: "Submit a generation request through the same path as the Generate view. Returns immediately; poll generation_status or history for completion.",
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
            name: "generation_status",
            title: "Generation Status",
            description: "Return the current generation state, latest completed result metadata, and current MCP operation when present.",
            properties: [:]
        ),
        tool(
            name: "editor_status",
            title: "Editor Status",
            description: "Return editor readiness, loaded audio metadata, supported editor actions, and the active editor operation when present.",
            properties: [:]
        ),
        tool(
            name: "load_editor_audio",
            title: "Load Editor Audio",
            description: "Load a history audio id or readable local audio path into the editor coordinator as a background operation. Returns an operationID for polling.",
            properties: [
                "id": [
                    "type": "string",
                    "description": "History entry id or audio resource id. Prefer this over path for sandbox-safe access."
                ],
                "historyID": [
                    "type": "string",
                    "description": "History entry id to load into the editor."
                ],
                "historyId": [
                    "type": "string",
                    "description": "Alias for historyID."
                ],
                "audioID": [
                    "type": "string",
                    "description": "Audio resource id to load into the editor."
                ],
                "audioId": [
                    "type": "string",
                    "description": "Alias for audioID."
                ],
                "path": [
                    "type": "string",
                    "description": "Readable local audio file path. History/audio ids are preferred because sandboxed app builds may not read arbitrary paths."
                ],
                "audioPath": [
                    "type": "string",
                    "description": "Alias for path."
                ],
                "filePath": [
                    "type": "string",
                    "description": "Alias for path."
                ],
                "originalPrompt": [
                    "type": "string",
                    "description": "Optional source prompt/context associated with the audio."
                ],
                "prompt": [
                    "type": "string",
                    "description": "Alias for originalPrompt."
                ],
                "dryRun": [
                    "type": "boolean",
                    "description": "Validate the local file path and echo the resolved request without loading audio."
                ],
                "validateOnly": [
                    "type": "boolean",
                    "description": "Alias for dryRun."
                ]
            ]
        ),
        tool(
            name: "submit_editor_inpaint",
            title: "Submit Editor Inpaint",
            description: "Submit a confirmed editor inpaint/repaint request against the loaded editor audio as a background operation. Use dryRun to validate without mutating audio.",
            properties: [
                "mode": [
                    "type": "string",
                    "enum": ["inpaint", "repaint"],
                    "description": "Editor edit mode. Defaults to inpaint."
                ],
                "editMode": [
                    "type": "string",
                    "enum": ["inpaint", "repaint"],
                    "description": "Alias for mode."
                ],
                "regionPrompt": [
                    "type": "string",
                    "description": "Prompt describing the replacement content for the selected region."
                ],
                "prompt": [
                    "type": "string",
                    "description": "Alias for regionPrompt."
                ],
                "originalPrompt": [
                    "type": "string",
                    "description": "Optional source/full-track context prompt. Defaults to loaded audio context when available."
                ],
                "sourcePrompt": [
                    "type": "string",
                    "description": "Alias for originalPrompt."
                ],
                "selection": [
                    "type": "object",
                    "additionalProperties": false,
                    "description": "Selected time range and optional frequency band.",
                    "properties": [
                        "startTime": ["type": "number"],
                        "endTime": ["type": "number"],
                        "start": ["type": "number"],
                        "end": ["type": "number"],
                        "frequencyBand": [
                            "type": "object",
                            "additionalProperties": false,
                            "properties": [
                                "lowCutHz": ["type": "number"],
                                "highCutHz": ["type": "number"],
                                "lowHz": ["type": "number"],
                                "highHz": ["type": "number"],
                                "low": ["type": "number"],
                                "high": ["type": "number"]
                            ]
                        ]
                    ]
                ],
                "selectionStart": [
                    "type": "number",
                    "description": "Selected region start time in seconds."
                ],
                "selectionEnd": [
                    "type": "number",
                    "description": "Selected region end time in seconds."
                ],
                "startTime": [
                    "type": "number",
                    "description": "Alias for selectionStart."
                ],
                "endTime": [
                    "type": "number",
                    "description": "Alias for selectionEnd."
                ],
                "start": [
                    "type": "number",
                    "description": "Alias for selectionStart."
                ],
                "end": [
                    "type": "number",
                    "description": "Alias for selectionEnd."
                ],
                "frequencyBand": [
                    "type": "object",
                    "additionalProperties": false,
                    "description": "Optional frequency-limited edit band in Hz.",
                    "properties": [
                        "lowCutHz": ["type": "number"],
                        "highCutHz": ["type": "number"],
                        "lowHz": ["type": "number"],
                        "highHz": ["type": "number"],
                        "low": ["type": "number"],
                        "high": ["type": "number"]
                    ]
                ],
                "lowCutHz": [
                    "type": "number",
                    "description": "Optional low frequency bound in Hz."
                ],
                "highCutHz": [
                    "type": "number",
                    "description": "Optional high frequency bound in Hz."
                ],
                "lowHz": [
                    "type": "number",
                    "description": "Alias for lowCutHz."
                ],
                "highHz": [
                    "type": "number",
                    "description": "Alias for highCutHz."
                ],
                "low": [
                    "type": "number",
                    "description": "Alias for lowCutHz."
                ],
                "high": [
                    "type": "number",
                    "description": "Alias for highCutHz."
                ],
                "preset": [
                    "type": "string",
                    "enum": ["subtle", "moderate", "full"],
                    "description": "Optional strength/guidance preset."
                ],
                "strength": [
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Denoising strength from 0.0 to 1.0. Defaults to 0.6."
                ],
                "denoiseStrength": [
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Alias for strength."
                ],
                "editStrength": [
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Alias for strength."
                ],
                "guidanceScale": [
                    "type": "number",
                    "exclusiveMinimum": 0,
                    "description": "Positive classifier-free guidance scale. Defaults to 7.0."
                ],
                "guidance": [
                    "type": "number",
                    "exclusiveMinimum": 0,
                    "description": "Alias for guidanceScale."
                ],
                "pipeline": [
                    "type": "string",
                    "enum": ["auto", "aceStep", "stableAudio"],
                    "description": "Generation pipeline to use for the edit. Defaults to auto."
                ],
                "pipelineOverride": [
                    "type": "string",
                    "enum": ["auto", "aceStep", "stableAudio"],
                    "description": "Alias for pipeline."
                ],
                "seed": [
                    "type": ["integer", "string", "null"],
                    "description": "Optional deterministic seed, or 'random'."
                ],
                "confirm": [
                    "type": "boolean",
                    "description": "Must be true for an actual editor mutation."
                ],
                "dryRun": [
                    "type": "boolean",
                    "description": "Validate and echo the resolved request without mutating editor audio."
                ],
                "validateOnly": [
                    "type": "boolean",
                    "description": "Alias for dryRun."
                ]
            ]
        ),
        tool(
            name: "render_editor_audio",
            title: "Render Editor Audio",
            description: "Render the loaded editor buffer or a selected time range to a WAV/AIFF file as a background operation. Use dryRun to resolve the output path without writing.",
            properties: [
                "format": [
                    "type": "string",
                    "enum": ["wav", "aiff"],
                    "description": "Lossless output format. Defaults to wav."
                ],
                "fileFormat": [
                    "type": "string",
                    "enum": ["wav", "aiff"],
                    "description": "Alias for format."
                ],
                "outputPath": [
                    "type": "string",
                    "description": "Optional output file path. The extension is normalized to the requested format. Sandboxed builds may reject arbitrary paths."
                ],
                "path": [
                    "type": "string",
                    "description": "Alias for outputPath."
                ],
                "outputDirectory": [
                    "type": "string",
                    "description": "Optional output directory. Defaults to MCPExports beside the loaded source audio."
                ],
                "directory": [
                    "type": "string",
                    "description": "Alias for outputDirectory."
                ],
                "fileName": [
                    "type": "string",
                    "description": "Optional file name when outputDirectory is used."
                ],
                "selection": [
                    "type": "object",
                    "additionalProperties": false,
                    "description": "Optional selected time range to render. Omit to render the full editor buffer.",
                    "properties": [
                        "startTime": ["type": "number"],
                        "endTime": ["type": "number"],
                        "start": ["type": "number"],
                        "end": ["type": "number"]
                    ]
                ],
                "selectionStart": [
                    "type": "number",
                    "description": "Optional selected region start time in seconds."
                ],
                "selectionEnd": [
                    "type": "number",
                    "description": "Optional selected region end time in seconds."
                ],
                "startTime": [
                    "type": "number",
                    "description": "Alias for selectionStart."
                ],
                "endTime": [
                    "type": "number",
                    "description": "Alias for selectionEnd."
                ],
                "start": [
                    "type": "number",
                    "description": "Alias for selectionStart."
                ],
                "end": [
                    "type": "number",
                    "description": "Alias for selectionEnd."
                ],
                "overwrite": [
                    "type": "boolean",
                    "description": "Allow replacing an existing output file. Defaults to false."
                ],
                "dryRun": [
                    "type": "boolean",
                    "description": "Validate and echo the resolved render request without writing a file."
                ],
                "validateOnly": [
                    "type": "boolean",
                    "description": "Alias for dryRun."
                ]
            ]
        ),
        tool(
            name: "list_operations",
            title: "List Operations",
            description: "Return recent MCP-observable operations with status, progress, result, error, and cancellation metadata.",
            properties: [
                "limit": [
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "description": "Maximum number of operation records to return. Defaults to 20."
                ]
            ]
        ),
        tool(
            name: "get_operation",
            title: "Get Operation",
            description: "Return one MCP-observable operation by id.",
            properties: [
                "id": [
                    "type": "string",
                    "description": "Operation id returned by generate, generate_song, load_editor_audio, submit_editor_inpaint, render_editor_audio, list_operations, editor_status, or generation_status."
                ],
                "operationID": [
                    "type": "string",
                    "description": "Alias for id."
                ],
                "operationId": [
                    "type": "string",
                    "description": "Alias for id."
                ]
            ]
        ),
        tool(
            name: "cancel_operation",
            title: "Cancel Operation",
            description: "Cancel a running MCP-observable operation by id. Generation and editor background operations use the same operation registry.",
            properties: [
                "id": [
                    "type": "string",
                    "description": "Operation id returned by generate, generate_song, load_editor_audio, submit_editor_inpaint, render_editor_audio, list_operations, editor_status, or generation_status."
                ],
                "operationID": [
                    "type": "string",
                    "description": "Alias for id."
                ],
                "operationId": [
                    "type": "string",
                    "description": "Alias for id."
                ],
                "reason": [
                    "type": "string",
                    "description": "Optional human-readable cancellation reason."
                ]
            ]
        ),
        tool(
            name: "release_memory",
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

    private static func sdkTool(_ definition: [String: Any]) -> Tool? {
        guard let name = definition["name"] as? String,
              let description = definition["description"] as? String else {
            return nil
        }

        let inputSchema = (try? MCPValueConverter.value(from: definition["inputSchema"] ?? [:]))
            ?? .object([:])
        return Tool(
            name: name,
            title: definition["title"] as? String,
            description: description,
            inputSchema: inputSchema
        )
    }
}
