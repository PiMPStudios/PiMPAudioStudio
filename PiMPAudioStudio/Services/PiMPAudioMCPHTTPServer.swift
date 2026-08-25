import Foundation
import Darwin
import MCP

final class MCPHTTPServer {
    private let port: UInt16
    private let bindHost: String
    private let allowNonLocalOrigins: Bool
    private let tokenProvider: @MainActor () -> String?
    private let onReady: @MainActor () -> Void
    private let onFailed: @MainActor (Error) -> Void
    private let handler: @MainActor (MCP.HTTPRequest) async -> MCP.HTTPResponse
    private let acceptQueue = DispatchQueue(label: "com.pimpstudios.PiMPAudioStudio.mcp.http.accept")
    private let clientQueue = DispatchQueue(label: "com.pimpstudios.PiMPAudioStudio.mcp.http.client", attributes: .concurrent)
    private let stateLock = NSLock()
    private var listenSocket: Int32 = -1
    /// Live client FDs keyed by identity generation. Generation prevents a
    /// stale async close from shutting down a later connection that reuses the FD.
    private var clientSockets: [Int32: UInt64] = [:]
    private var nextClientGeneration: UInt64 = 1
    private var isRunning = false

    init(
        port: UInt16,
        bindHost: String = "127.0.0.1",
        allowNonLocalOrigins: Bool = false,
        tokenProvider: @escaping @MainActor () -> String?,
        onReady: @escaping @MainActor () -> Void,
        onFailed: @escaping @MainActor (Error) -> Void,
        handler: @escaping @MainActor (MCP.HTTPRequest) async -> MCP.HTTPResponse
    ) throws {
        self.port = port
        self.bindHost = bindHost
        self.allowNonLocalOrigins = allowNonLocalOrigins
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
            try bindAddress(socketFD)
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

    /// Constant-time compare so LAN-exposed tokens do not leak via early exit.
    static func tokensMatch(_ lhs: String, _ rhs: String) -> Bool {
        let left = Array(lhs.utf8)
        let right = Array(rhs.utf8)
        var diff: UInt8 = left.count == right.count ? 0 : 1
        let count = max(left.count, right.count)
        for index in 0..<count {
            let a = index < left.count ? left[index] : 0
            let b = index < right.count ? right[index] : 0
            diff |= a ^ b
        }
        return diff == 0
    }

    func stop() {
        stateLock.lock()
        isRunning = false
        let socketFD = listenSocket
        listenSocket = -1
        // Take ownership of every live client; later closeClient(generation:)
        // calls for these identities become no-ops.
        let clients = clientSockets
        clientSockets.removeAll()
        stateLock.unlock()

        if socketFD >= 0 {
            Darwin.shutdown(socketFD, SHUT_RDWR)
            Darwin.close(socketFD)
        }
        for (client, _) in clients {
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

    private func bindAddress(_ socketFD: Int32) throws {
        var address = sockaddr_in()
        address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        address.sin_family = sa_family_t(AF_INET)
        address.sin_port = in_port_t(port).bigEndian
        let resolvedHost = Self.normalizedBindHost(bindHost)
        // inet_addr returns network-order IPv4. "0.0.0.0" resolves to INADDR_ANY (0).
        // Failure is INADDR_NONE / (in_addr_t)-1.
        let addressValue = inet_addr(resolvedHost)
        if addressValue == in_addr_t(bitPattern: -1), resolvedHost != "255.255.255.255" {
            throw MCPHTTPServerError(
                operation: "bind \(resolvedHost):\(port)",
                errnoCode: EINVAL
            )
        }
        address.sin_addr = in_addr(s_addr: addressValue)

        let result = withUnsafePointer(to: &address) { pointer in
            pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) { sockaddrPointer in
                Darwin.bind(socketFD, sockaddrPointer, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        guard result == 0 else {
            throw Self.posixError("bind \(resolvedHost):\(port)")
        }
    }

    static func normalizedBindHost(_ host: String) -> String {
        let trimmed = host.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        switch trimmed {
        case "", "localhost", "loopback", "local":
            return "127.0.0.1"
        case "0.0.0.0", "*", "any", "all", "allinterfaces", "all-interfaces":
            return "0.0.0.0"
        default:
            return trimmed
        }
    }

    static func isRetryableAcceptError(_ errnoCode: Int32) -> Bool {
        errnoCode == EINTR
            || errnoCode == EAGAIN
            || errnoCode == EWOULDBLOCK
            || errnoCode == EMFILE
            || errnoCode == ENFILE
            || errnoCode == ECONNABORTED
    }

    private func acceptLoop(_ socketFD: Int32) {
        var backoffNanoseconds: UInt64 = 20_000_000
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
                if !isServerRunning(socketFD) {
                    return
                }
                if Self.isRetryableAcceptError(errnoCode) {
                    if errnoCode != EINTR {
                        nanosleepBackoff(&backoffNanoseconds)
                    }
                    continue
                }
                let error = Self.posixError("accept", errnoCode: errnoCode)
                Task { @MainActor in
                    self.onFailed(error)
                }
                return
            }

            backoffNanoseconds = 20_000_000
            configureClientSocket(client)
            guard let generation = registerClient(client) else {
                // Server stopped between accept and registration.
                Darwin.shutdown(client, SHUT_RDWR)
                Darwin.close(client)
                continue
            }
            clientQueue.async { [weak self] in
                self?.receive(on: client, generation: generation, buffer: Data())
            }
        }
    }

    private func nanosleepBackoff(_ backoffNanoseconds: inout UInt64) {
        let capped = min(backoffNanoseconds, 1_000_000_000)
        var spec = timespec(
            tv_sec: capped == 1_000_000_000 ? 1 : 0,
            tv_nsec: capped == 1_000_000_000 ? 0 : Int(capped)
        )
        nanosleep(&spec, nil)
        backoffNanoseconds = min(backoffNanoseconds * 2, 1_000_000_000)
    }

    private func configureClientSocket(_ client: Int32) {
        // Slow or large JSON-RPC bodies over this socket are rejected after 10s;
        // that matches interactive MCP clients, not bulk uploads.
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

    /// Registers a client FD and returns its close-generation identity.
    /// Returns `nil` when the server is no longer running.
    private func registerClient(_ client: Int32) -> UInt64? {
        stateLock.lock()
        defer { stateLock.unlock() }
        guard isRunning else { return nil }
        let generation = nextClientGeneration
        nextClientGeneration &+= 1
        clientSockets[client] = generation
        return generation
    }

    /// Idempotent close: only the owner of `(fd, generation)` shuts down the socket.
    /// Safe when `stop()` already closed the client, or when the FD has been reused.
    private func closeClient(_ client: Int32, generation: UInt64) {
        stateLock.lock()
        guard clientSockets[client] == generation else {
            stateLock.unlock()
            return
        }
        clientSockets.removeValue(forKey: client)
        stateLock.unlock()

        Darwin.shutdown(client, SHUT_RDWR)
        Darwin.close(client)
    }

    private func receive(on client: Int32, generation: UInt64, buffer: Data) {
        var buffer = buffer
        var chunk = [UInt8](repeating: 0, count: 64 * 1024)
        let maxRequestBytes = 2 * 1024 * 1024

        while buffer.count < maxRequestBytes {
            // Drop out if stop() already claimed this connection.
            if !isClientLive(client, generation: generation) {
                return
            }

            let capacity = chunk.count
            let byteCount = chunk.withUnsafeMutableBytes { bytes in
                Darwin.recv(client, bytes.baseAddress, capacity, 0)
            }

            if byteCount > 0 {
                buffer.append(contentsOf: chunk.prefix(byteCount))
                if let request = RawHTTPRequest(data: buffer) {
                    Task {
                        let response = await self.respond(to: request)
                        await self.send(response, on: client, generation: generation)
                        self.closeClient(client, generation: generation)
                    }
                    return
                }
                continue
            }

            if byteCount == 0 {
                closeClient(client, generation: generation)
                return
            }

            if errno == EAGAIN || errno == EWOULDBLOCK {
                sendAndClose(.badRequest(), on: client, generation: generation)
                return
            }

            closeClient(client, generation: generation)
            return
        }

        sendAndClose(.badRequest(), on: client, generation: generation)
    }

    private func isClientLive(_ client: Int32, generation: UInt64) -> Bool {
        stateLock.lock()
        let live = clientSockets[client] == generation
        stateLock.unlock()
        return live
    }

    private func respond(to request: RawHTTPRequest) async -> RawHTTPResponse {
        let corsOrigin = request.corsAllowedOrigin(allowNonLocalOrigins: allowNonLocalOrigins)
        guard request.hasTrustedOrigin(allowNonLocalOrigins: allowNonLocalOrigins) else {
            return .jsonError(statusCode: 403, message: "Forbidden Origin", corsOrigin: corsOrigin)
        }

        if request.method == "OPTIONS" {
            return .empty(statusCode: 204, statusText: "No Content", corsOrigin: corsOrigin)
        }

        if request.method == "GET", request.path == "/health" {
            return .json([
                "status": "ok",
                "endpoint": "/mcp"
            ], corsOrigin: corsOrigin)
        }

        guard request.path == "/mcp" else {
            return .jsonError(statusCode: 404, message: "MCP endpoint is /mcp", corsOrigin: corsOrigin)
        }

        let expectedToken = await tokenProvider()
        if let expectedToken, !expectedToken.isEmpty {
            let headerToken = request.bearerToken ?? ""
            guard Self.tokensMatch(headerToken, expectedToken) else {
                return .jsonError(
                    statusCode: 401,
                    message: "Missing or invalid MCP token",
                    corsOrigin: corsOrigin
                )
            }
        }

        let sdkResponse = await handler(request.sdkRequest)
        return .sdk(sdkResponse, corsOrigin: corsOrigin)
    }

    private func send(
        _ response: RawHTTPResponse,
        on client: Int32,
        generation: UInt64
    ) async {
        guard isClientLive(client, generation: generation) else { return }
        switch response {
        case .fixed(let data):
            _ = sendData(data, on: client, generation: generation)
        case .stream(let header, let stream):
            guard sendData(header, on: client, generation: generation) else { return }
            do {
                for try await chunk in stream {
                    guard sendData(chunk, on: client, generation: generation) else { return }
                }
            } catch {
                return
            }
        }
    }

    private func sendAndClose(
        _ response: RawHTTPResponse,
        on client: Int32,
        generation: UInt64
    ) {
        Task {
            await self.send(response, on: client, generation: generation)
            self.closeClient(client, generation: generation)
        }
    }

    private func sendData(
        _ data: Data,
        on client: Int32,
        generation: UInt64
    ) -> Bool {
        guard isClientLive(client, generation: generation) else { return false }
        if data.isEmpty { return true }
        return data.withUnsafeBytes { pointer in
            guard var address = pointer.baseAddress, pointer.count > 0 else { return true }
            var remaining = pointer.count
            while remaining > 0 {
                guard isClientLive(client, generation: generation) else { return false }
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

struct RawHTTPRequest {
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

        let bodyStart = headerEnd
        let availableBodyBytes = data.count - bodyStart
        guard let contentLength = Int(headers["content-length"] ?? "0"),
              contentLength >= 0,
              contentLength <= availableBodyBytes else {
            return nil
        }

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

    func hasTrustedOrigin(allowNonLocalOrigins: Bool) -> Bool {
        guard let origin = headers["origin"]?.trimmingCharacters(in: .whitespacesAndNewlines),
              !origin.isEmpty else {
            return true
        }
        // Network-bound servers may receive LAN browser Origins; agent clients
        // usually omit Origin entirely.
        if allowNonLocalOrigins {
            return true
        }
        guard let url = URL(string: origin),
              let host = url.host?.lowercased() else {
            return false
        }
        return host == "localhost" || host == "127.0.0.1" || host == "::1"
    }

    /// Echo a specific Origin when the browser sent one we trust. Never `*`.
    /// Agent clients omit Origin and do not need CORS headers.
    func corsAllowedOrigin(allowNonLocalOrigins: Bool) -> String? {
        guard let origin = headers["origin"]?.trimmingCharacters(in: .whitespacesAndNewlines),
              !origin.isEmpty else {
            return nil
        }
        guard hasTrustedOrigin(allowNonLocalOrigins: allowNonLocalOrigins) else {
            return nil
        }
        return origin
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

    static func json(
        _ object: Any,
        statusCode: Int = 200,
        statusText: String = "OK",
        corsOrigin: String? = nil
    ) -> RawHTTPResponse {
        let body: Data
        if JSONSerialization.isValidJSONObject(object),
           let encoded = try? JSONSerialization.data(withJSONObject: object, options: [.prettyPrinted, .sortedKeys]) {
            body = encoded
        } else {
            body = Data("{\"error\":\"Invalid JSON response\"}".utf8)
        }
        return response(
            statusCode: statusCode,
            statusText: statusText,
            body: body,
            contentType: "application/json",
            corsOrigin: corsOrigin
        )
    }

    static func jsonError(
        statusCode: Int,
        message: String,
        corsOrigin: String? = nil
    ) -> RawHTTPResponse {
        json(["error": message], statusCode: statusCode, statusText: message, corsOrigin: corsOrigin)
    }

    static func badRequest() -> RawHTTPResponse {
        jsonError(statusCode: 400, message: "Bad Request")
    }

    static func empty(
        statusCode: Int,
        statusText: String,
        corsOrigin: String? = nil
    ) -> RawHTTPResponse {
        response(
            statusCode: statusCode,
            statusText: statusText,
            body: Data(),
            contentType: "application/json",
            corsOrigin: corsOrigin
        )
    }

    static func sdk(_ response: MCP.HTTPResponse, corsOrigin: String? = nil) -> RawHTTPResponse {
        if case .stream(let stream, let headers) = response {
            let contentType = headers[MCP.HTTPHeaderName.contentType] ?? "text/event-stream"
            return self.stream(
                statusCode: response.statusCode,
                statusText: statusText(for: response.statusCode),
                contentType: contentType,
                body: stream,
                extraHeaders: headers,
                corsOrigin: corsOrigin
            )
        }

        let body = response.bodyData ?? Data()
        let contentType = response.headers[MCP.HTTPHeaderName.contentType] ?? "application/json"
        return self.response(
            statusCode: response.statusCode,
            statusText: statusText(for: response.statusCode),
            body: body,
            contentType: contentType,
            extraHeaders: response.headers,
            corsOrigin: corsOrigin
        )
    }

    private static func stream(
        statusCode: Int,
        statusText: String,
        contentType: String,
        body: AsyncThrowingStream<Data, Error>,
        extraHeaders: [String: String] = [:],
        corsOrigin: String? = nil
    ) -> RawHTTPResponse {
        let headers = responseHeaders(
            contentType: contentType,
            contentLength: nil,
            // Stream is always followed by closeClient; advertise close so
            // clients do not try to reuse the socket after the body.
            connection: "close",
            extraHeaders: extraHeaders,
            corsOrigin: corsOrigin
        )
        return .stream(header: headerData(statusCode: statusCode, statusText: statusText, headers: headers), body: body)
    }

    private static func response(
        statusCode: Int,
        statusText: String,
        body: Data,
        contentType: String,
        extraHeaders: [String: String] = [:],
        corsOrigin: String? = nil
    ) -> RawHTTPResponse {
        let headers = responseHeaders(
            contentType: contentType,
            contentLength: "\(body.count)",
            connection: "close",
            extraHeaders: extraHeaders,
            corsOrigin: corsOrigin
        )
        var data = headerData(statusCode: statusCode, statusText: statusText, headers: headers)
        data.append(body)
        return .fixed(data)
    }

    private static func responseHeaders(
        contentType: String,
        contentLength: String?,
        connection: String,
        extraHeaders: [String: String],
        corsOrigin: String? = nil
    ) -> [String: String] {
        var headers = [
            "Content-Type": contentType,
            "Connection": connection,
            "MCP-Protocol-Version": "2025-06-18",
            "Allow": "POST, GET, DELETE, OPTIONS"
        ]
        if let contentLength {
            headers["Content-Length"] = contentLength
        }
        headers.merge(extraHeaders) { _, sdkValue in sdkValue }
        if let corsOrigin {
            headers["Access-Control-Allow-Origin"] = corsOrigin
            headers["Access-Control-Allow-Headers"] =
                "Accept, Content-Type, Authorization, Mcp-Method, MCP-Protocol-Version, MCP-Session-Id, Last-Event-ID"
            headers["Access-Control-Allow-Methods"] = "POST, GET, DELETE, OPTIONS"
            headers["Vary"] = "Origin"
        } else {
            headers.removeValue(forKey: "Access-Control-Allow-Origin")
            headers.removeValue(forKey: "Access-Control-Allow-Headers")
            headers.removeValue(forKey: "Access-Control-Allow-Methods")
        }
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
