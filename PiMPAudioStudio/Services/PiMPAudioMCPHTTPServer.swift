import Foundation
import Darwin
import MCP

final class MCPHTTPServer {
    private let port: UInt16
    private let tokenProvider: @MainActor () -> String?
    private let onReady: @MainActor () -> Void
    private let onFailed: @MainActor (Error) -> Void
    private let handler: @MainActor (MCP.HTTPRequest) async -> MCP.HTTPResponse
    private let acceptQueue = DispatchQueue(label: "com.pimpstudios.PiMPAudioStudio.mcp.http.accept")
    private let clientQueue = DispatchQueue(label: "com.pimpstudios.PiMPAudioStudio.mcp.http.client", attributes: .concurrent)
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
