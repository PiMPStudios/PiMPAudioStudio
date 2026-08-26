import Foundation
import Darwin
import MCP
import OSLog

enum PiMPAudioMCPServerState: Equatable {
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

enum MCPBindAddress: String, CaseIterable, Identifiable, Sendable {
    case loopback = "127.0.0.1"
    case allInterfaces = "0.0.0.0"

    var id: String { rawValue }

    var title: String {
        switch self {
        case .loopback:
            return "Localhost (127.0.0.1)"
        case .allInterfaces:
            return "All interfaces (0.0.0.0)"
        }
    }

    var detail: String {
        switch self {
        case .loopback:
            return "Only this Mac can connect."
        case .allInterfaces:
            return "Other machines on your network can connect. Use a bearer token."
        }
    }

    var isLoopbackOnly: Bool { self == .loopback }
}

enum MCPConfirmationDecision: Equatable {
    case proceed
    case elicit
    case declined(action: String, reason: String)

    static func evaluate(
        requiresConfirmation: Bool,
        confirmArgument: Bool?,
        isDryRun: Bool,
        clientSupportsFormElicitation: Bool
    ) -> MCPConfirmationDecision {
        guard requiresConfirmation, confirmArgument != true, !isDryRun else {
            return .proceed
        }
        if clientSupportsFormElicitation {
            return .elicit
        }
        return .declined(
            action: "missingConfirm",
            reason: "This tool requires confirm: true or MCP form elicitation approval."
        )
    }
}

@MainActor
final class PiMPAudioMCPServer: ObservableObject {
    private static let enabledKey = AppRuntimeIdentity.defaultsKey("mcp.enabled")
    private static let tokenRequiredKey = AppRuntimeIdentity.defaultsKey("mcp.tokenRequired")
    private static let tokenKey = AppRuntimeIdentity.defaultsKey("mcp.token")
    private static let tokenAccount = "mcp-bearer-token"
    private static let bindAddressKey = AppRuntimeIdentity.defaultsKey("mcp.bindAddress")
    private static let portKey = AppRuntimeIdentity.defaultsKey("mcp.port")
    static let defaultPort: UInt16 = 38765
    static let minimumPort: UInt16 = 1_024
    static let maximumPort: UInt16 = 65_535

    @Published private(set) var isEnabled: Bool
    @Published private(set) var isTokenRequired: Bool
    @Published private(set) var token: String
    @Published private(set) var bindAddress: MCPBindAddress
    @Published private(set) var port: UInt16
    @Published private(set) var state: PiMPAudioMCPServerState = .stopped

    private let logger = Logger(subsystem: "com.pimpstudios.PiMPAudioStudio", category: "MCP")
    private let facade: PiMPAudioCommandFacade
    private var httpServer: MCPHTTPServer?
    private var sdkBridge: PiMPAudioMCPSDKBridge?
    private var startupTask: Task<Void, Never>?

    /// Preferred client URL for this Mac (always uses 127.0.0.1 for local clients).
    var endpoint: String {
        "http://127.0.0.1:\(port)/mcp"
    }

    /// Human-readable bind summary, e.g. `127.0.0.1:38765` or `0.0.0.0:38765`.
    var bindDescription: String {
        "\(bindAddress.rawValue):\(port)"
    }

    /// Suggested URL for LAN clients when bound to all interfaces.
    var networkEndpoint: String? {
        guard bindAddress == .allInterfaces,
              let host = Self.primaryLANAddress() else {
            return nil
        }
        return "http://\(host):\(port)/mcp"
    }

    var isNetworkExposed: Bool {
        bindAddress == .allInterfaces
    }

    init(facade: PiMPAudioCommandFacade) {
        self.facade = facade
        let defaults = UserDefaults.standard
        self.isEnabled = defaults.bool(forKey: Self.enabledKey)
        let bindAddress = Self.loadBindAddress(from: defaults)
        self.bindAddress = bindAddress
        self.port = Self.loadPort(from: defaults)
        let requestedToken = defaults.bool(forKey: Self.tokenRequiredKey)
        let isTokenRequired = Self.tokenIsRequired(
            requested: requestedToken,
            bindAddress: bindAddress
        )
        self.isTokenRequired = isTokenRequired
        if isTokenRequired != requestedToken {
            defaults.set(true, forKey: Self.tokenRequiredKey)
        }
        let defaultsToken = defaults.string(forKey: Self.tokenKey)
        let keychainToken = KeychainStringStore.isAvailable
            ? KeychainStringStore.read(account: Self.tokenAccount)
            : nil
        let resolved = MCPTokenResolution.resolve(
            keychainValue: keychainToken,
            defaultsValue: defaultsToken,
            generated: Self.generateToken()
        )
        self.token = resolved.token
        Self.persistToken(resolved)

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
        let enforced = Self.tokenIsRequired(requested: required, bindAddress: bindAddress)
        UserDefaults.standard.set(enforced, forKey: Self.tokenRequiredKey)
        isTokenRequired = enforced
        if enforced && token.isEmpty {
            regenerateToken()
        }
    }

    func setBindAddress(_ address: MCPBindAddress) {
        let normalized = MCPBindAddress(rawValue: MCPHTTPServer.normalizedBindHost(address.rawValue))
            ?? address
        guard normalized != bindAddress else { return }
        bindAddress = normalized
        UserDefaults.standard.set(normalized.rawValue, forKey: Self.bindAddressKey)
        // Network exposure without a token is unsafe; the toggle cannot turn it off.
        if normalized == .allInterfaces {
            setTokenRequired(true)
        }
        restartIfEnabled()
    }

    nonisolated static func tokenIsRequired(requested: Bool, bindAddress: MCPBindAddress) -> Bool {
        requested || bindAddress == .allInterfaces
    }

    nonisolated static func toolRequiresElicitableConfirmation(_ toolName: String) -> Bool {
        switch toolName {
        case "generate",
             "generate_song",
             "write_lyrics",
             "apply_adapter",
             "load_editor_audio",
             "render_editor_audio",
             "mark_preferred_take",
             "cancel_generation",
             "cancel_operation",
             "submit_editor_inpaint",
             "apply_editor_effect_chain",
             "start_training",
             "stop_training",
             "launch_runpod_pod",
             "stop_runpod_pod",
             "create_remote_training_job",
             "upload_remote_training_job",
             "start_remote_training_job",
             "pull_remote_training_job",
             "build_training_dataset",
             "enrich_training_dataset",
             "update_training_dataset_review",
             "import_adapter",
             "delete_adapter",
             "merge_adapters",
             "download_model",
             "set_active_model",
             "delete_history_entries",
             "save_generation_preset",
             "delete_generation_preset",
             "run_adapter_audition",
             "release_memory":
            return true
        default:
            return false
        }
    }

    @discardableResult
    func setPort(_ newPort: UInt16) -> Bool {
        guard Self.isValidPort(newPort) else { return false }
        guard newPort != port else { return true }
        port = newPort
        UserDefaults.standard.set(Int(newPort), forKey: Self.portKey)
        restartIfEnabled()
        return true
    }

    @discardableResult
    func setPort(from text: String) -> Bool {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let value = UInt16(trimmed), setPort(value) else {
            return false
        }
        return true
    }

    func regenerateToken() {
        let generated = Self.generateToken()
        token = generated
        Self.persistToken(
            MCPTokenResolution(token: generated, writeToKeychain: generated, clearDefaults: true)
        )
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

    private func restartIfEnabled() {
        guard isEnabled else { return }
        stop()
        start()
    }

    private func start() {
        guard httpServer == nil, startupTask == nil else {
            state = .running(endpoint: endpoint)
            return
        }

        state = .starting
        let bindHost = bindAddress.rawValue
        let listenPort = port
        let allowNonLocalOrigins = !bindAddress.isLoopbackOnly
        startupTask = Task { @MainActor [weak self] in
            guard let self else { return }

            let bridge = PiMPAudioMCPSDKBridge(
                facade: self.facade,
                version: Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "0.8.0"
            )
            var startedServer: MCPHTTPServer?

            do {
                try Task.checkCancellation()
                try await bridge.start()
                try Task.checkCancellation()

                let server = try MCPHTTPServer(
                    port: listenPort,
                    bindHost: bindHost,
                    allowNonLocalOrigins: allowNonLocalOrigins,
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
        // Leave UserDefaults enabled as-is so a transient accept/bind failure
        // retries on the next launch instead of silently staying off.
    }

    private static func generateToken() -> String {
        UUID().uuidString.replacingOccurrences(of: "-", with: "").lowercased()
    }

    private static func persistToken(_ resolved: MCPTokenResolution) {
        if KeychainStringStore.isAvailable {
            if let write = resolved.writeToKeychain {
                do {
                    try KeychainStringStore.write(account: tokenAccount, value: write)
                } catch {
                    Logger(
                        subsystem: "com.pimpstudios.PiMPAudioStudio",
                        category: "MCP"
                    ).error(
                        "Failed to persist MCP token in Keychain: \(error.localizedDescription, privacy: .public)"
                    )
                    return
                }
            }
            if resolved.clearDefaults {
                UserDefaults.standard.removeObject(forKey: tokenKey)
            }
            return
        }
        UserDefaults.standard.set(resolved.token, forKey: tokenKey)
    }

    static func isValidPort(_ port: UInt16) -> Bool {
        port >= minimumPort && port <= maximumPort
    }

    private static func loadBindAddress(from defaults: UserDefaults) -> MCPBindAddress {
        let raw = defaults.string(forKey: bindAddressKey) ?? MCPBindAddress.loopback.rawValue
        let normalized = MCPHTTPServer.normalizedBindHost(raw)
        return MCPBindAddress(rawValue: normalized) ?? .loopback
    }

    private static func loadPort(from defaults: UserDefaults) -> UInt16 {
        guard defaults.object(forKey: portKey) != nil else {
            return defaultPort
        }
        let value = defaults.integer(forKey: portKey)
        guard value >= Int(minimumPort), value <= Int(maximumPort) else {
            return defaultPort
        }
        return UInt16(value)
    }

    /// Best-effort primary IPv4 address for LAN endpoint display.
    private static func primaryLANAddress() -> String? {
        var ifaddrPointer: UnsafeMutablePointer<ifaddrs>?
        guard getifaddrs(&ifaddrPointer) == 0, let first = ifaddrPointer else {
            return nil
        }
        defer { freeifaddrs(ifaddrPointer) }

        var candidate: String?
        var pointer: UnsafeMutablePointer<ifaddrs>? = first
        while let interface = pointer {
            defer { pointer = interface.pointee.ifa_next }
            let flags = Int32(interface.pointee.ifa_flags)
            let isUp = (flags & IFF_UP) != 0
            let isLoopback = (flags & IFF_LOOPBACK) != 0
            guard isUp, !isLoopback else { continue }
            guard let addr = interface.pointee.ifa_addr, addr.pointee.sa_family == UInt8(AF_INET) else {
                continue
            }

            var hostname = [CChar](repeating: 0, count: Int(NI_MAXHOST))
            let result = getnameinfo(
                addr,
                socklen_t(addr.pointee.sa_len),
                &hostname,
                socklen_t(hostname.count),
                nil,
                0,
                NI_NUMERICHOST
            )
            guard result == 0 else { continue }
            let ip = String(cString: hostname)
            // Prefer common private LAN ranges.
            if ip.hasPrefix("192.168.") || ip.hasPrefix("10.") || ip.hasPrefix("172.") {
                return ip
            }
            if candidate == nil {
                candidate = ip
            }
        }
        return candidate
    }

}

@MainActor
private final class PiMPAudioMCPSDKBridge {
    private let facade: PiMPAudioCommandFacade
    private let version: String
    private var server: Server
    private var transport: StatefulHTTPServerTransport
    private var isStarted = false
    private var hasActiveSession = false
    private var activeSessionID: String?
    private var clientSupportsFormElicitation = false
    private var clientSupportsRoots = false
    private var clientRootsListChanged = false
    private var clientRoots: [Root] = []
    private var clientRootsRefreshTask: Task<Void, Never>?
    private var isRefreshingClientRoots = false
    private var clientRootsRefreshedAt: Date?
    private var clientRootsLastError: String?

    init(facade: PiMPAudioCommandFacade, version: String) {
        self.facade = facade
        self.version = version
        let components = Self.makeServer(version: version)
        self.server = components.server
        self.transport = components.transport
    }

    private static func makeServer(version: String) -> (server: Server, transport: StatefulHTTPServerTransport) {
        let server = Server(
            name: AppRuntimeIdentity.displayName,
            version: version,
            capabilities: .init(
                completions: .init(),
                prompts: .init(listChanged: false),
                resources: .init(subscribe: false, listChanged: false),
                tools: .init(listChanged: false)
            )
        )
        return (server, StatefulHTTPServerTransport())
    }

    func start() async throws {
        guard !isStarted else { return }
        await registerHandlers()
        try await server.start(transport: transport) { [weak self] _, capabilities in
            await self?.recordClientCapabilities(capabilities)
        }
        isStarted = true
    }

    func stop() async {
        guard isStarted else { return }
        await server.stop()
        isStarted = false
        resetClientSessionState()
    }

    func handle(_ request: MCP.HTTPRequest) async -> MCP.HTTPResponse {
        let isInitialize = isInitializeRequest(request)
        if isInitialize, hasActiveSession {
            if let resetError = await resetSessionTransport() {
                return resetError
            }
        }

        let response = await transport.handleRequest(request)

        if isInitialize {
            if response.statusCode == 200 {
                recordActiveSession(from: response)
            } else if isAlreadyInitializedResponse(response) {
                if let resetError = await resetSessionTransport() {
                    return resetError
                }
                let retryResponse = await transport.handleRequest(request)
                if retryResponse.statusCode == 200 {
                    recordActiveSession(from: retryResponse)
                }
                return retryResponse
            }
        } else if request.method.uppercased() == "DELETE", (200..<300).contains(response.statusCode) {
            let previousSessionID = activeSessionID ?? request.header(MCP.HTTPHeaderName.sessionID)
            if let resetError = await resetSessionTransport() {
                return resetError
            }
            return response.statusCode == 200 ? response : .ok(headers: sessionHeaders(for: previousSessionID))
        } else if request.method.uppercased() == "DELETE", shouldForceResetAfterDelete(request: request, response: response) {
            let previousSessionID = activeSessionID ?? request.header(MCP.HTTPHeaderName.sessionID)
            if let resetError = await resetSessionTransport() {
                return resetError
            }
            return .ok(headers: sessionHeaders(for: previousSessionID))
        }

        return response
    }

    private func recordActiveSession(from response: MCP.HTTPResponse) {
        hasActiveSession = true
        activeSessionID = Self.headerValue(MCP.HTTPHeaderName.sessionID, in: response.headers)
    }

    private func shouldForceResetAfterDelete(request: MCP.HTTPRequest, response: MCP.HTTPResponse) -> Bool {
        guard hasActiveSession else { return false }
        let requestSessionID = request.header(MCP.HTTPHeaderName.sessionID)
        guard requestSessionID == nil || requestSessionID == activeSessionID else { return false }
        return response.statusCode == 400 || response.statusCode == 404
    }

    private func isAlreadyInitializedResponse(_ response: MCP.HTTPResponse) -> Bool {
        guard response.statusCode == 400,
              let body = response.bodyData,
              let text = String(data: body, encoding: .utf8) else {
            return false
        }
        return text.contains("Session already initialized")
            || text.contains("Server is already initialized")
    }

    private func sessionHeaders(for sessionID: String?) -> [String: String] {
        guard let sessionID, !sessionID.isEmpty else { return [:] }
        return [MCP.HTTPHeaderName.sessionID: sessionID]
    }

    private static func headerValue(_ name: String, in headers: [String: String]) -> String? {
        let lowercased = name.lowercased()
        return headers.first { $0.key.lowercased() == lowercased }?.value
    }

    private func resetSessionTransport() async -> MCP.HTTPResponse? {
        if isStarted {
            await server.stop()
            isStarted = false
        }

        let components = Self.makeServer(version: version)
        server = components.server
        transport = components.transport
        resetClientSessionState()

        do {
            await registerHandlers()
            try await server.start(transport: transport) { [weak self] _, capabilities in
                await self?.recordClientCapabilities(capabilities)
            }
            isStarted = true
            return nil
        } catch {
            return .error(statusCode: 500, .internalError("Failed to reset MCP session transport: \(error.localizedDescription)"))
        }
    }

    private func recordClientCapabilities(_ capabilities: Client.Capabilities) {
        clientSupportsFormElicitation = capabilities.elicitation?.form != nil
        clientSupportsRoots = capabilities.roots != nil
        clientRootsListChanged = capabilities.roots?.listChanged == true
        clientRoots = []
        clientRootsRefreshedAt = nil
        clientRootsLastError = nil

        if clientSupportsRoots {
            scheduleClientRootsRefresh(reason: "initialize")
        }
    }

    private func resetClientSessionState() {
        hasActiveSession = false
        activeSessionID = nil
        clientSupportsFormElicitation = false
        clientSupportsRoots = false
        clientRootsListChanged = false
        clientRoots = []
        clientRootsRefreshedAt = nil
        clientRootsLastError = nil
        isRefreshingClientRoots = false
        clientRootsRefreshTask?.cancel()
        clientRootsRefreshTask = nil
    }

    private func scheduleClientRootsRefresh(reason: String) {
        clientRootsRefreshTask?.cancel()
        clientRootsRefreshTask = Task { @MainActor [weak self] in
            try? await Task.sleep(for: .milliseconds(100))
            guard let self, !Task.isCancelled else { return }
            await self.refreshClientRoots(reason: reason)
        }
    }

    private func refreshClientRoots(reason: String) async {
        guard clientSupportsRoots else {
            clientRoots = []
            clientRootsLastError = nil
            clientRootsRefreshedAt = nil
            return
        }

        isRefreshingClientRoots = true
        defer {
            isRefreshingClientRoots = false
            clientRootsRefreshTask = nil
        }

        do {
            let roots = try await server.listRoots()
            guard !Task.isCancelled else { return }
            clientRoots = roots
            clientRootsLastError = nil
            clientRootsRefreshedAt = Date()
        } catch {
            guard !Task.isCancelled else { return }
            clientRoots = []
            clientRootsLastError = "\(reason): \(error.localizedDescription)"
            clientRootsRefreshedAt = Date()
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

        await server.withMethodHandler(ListPrompts.self) { _ in
            .init(prompts: MCPPromptCatalog.prompts, nextCursor: nil)
        }

        await server.withMethodHandler(GetPrompt.self) { params in
            try MCPPromptCatalog.getPrompt(params)
        }

        await server.withMethodHandler(Complete.self) { [weak self] params in
            guard let self else {
                return .init(completion: .init(values: []))
            }
            return await self.complete(params)
        }

        await server.withMethodHandler(ListResources.self) { [weak self] _ in
            guard let self else {
                return .init(resources: [], nextCursor: nil)
            }
            var resources = await self.facade.listResources()
            resources.append(MCPResourceCatalog.clientRootsResourceDefinition)
            return .init(resources: resources.compactMap(MCPResourceCatalog.sdkResource), nextCursor: nil)
        }

        await server.withMethodHandler(ListResourceTemplates.self) { _ in
            .init(templates: MCPResourceCatalog.templates, nextCursor: nil)
        }

        await server.withMethodHandler(ReadResource.self) { [weak self] params in
            guard let self else {
                throw MCPError.internalError("MCP server is shutting down")
            }
            return try await self.readResource(params)
        }

        await server.onNotification(RootsListChangedNotification.self) { [weak self] _ in
            await self?.scheduleClientRootsRefresh(reason: RootsListChangedNotification.name)
        }
    }

    private func complete(_ params: Complete.Parameters) async -> Complete.Result {
        let values: [String]

        switch params.ref {
        case .prompt(let promptRef):
            values = await promptCompletionValues(
                promptName: promptRef.name,
                argumentName: params.argument.name
            )
        case .resource(let resourceRef):
            values = await resourceCompletionValues(
                resourceURI: resourceRef.uri,
                argumentName: params.argument.name
            )
        }

        return completionResult(values, matching: params.argument.value)
    }

    private func promptCompletionValues(
        promptName: String,
        argumentName: String
    ) async -> [String] {
        switch argumentName {
        case "takeID":
            return historyCompletionValues()
        case "regionStart":
            return await currentRegionCompletionValues(key: "startTime", defaults: ["0", "5", "10", "15", "30"])
        case "regionEnd":
            return await currentRegionCompletionValues(key: "endTime", defaults: ["5", "10", "15", "30", "60"])
        case "variantCount":
            return ["2", "3", "4", "5"]
        case "duration":
            return ["5", "10", "15", "30", "60", "90", "120"]
        case "direction":
            return directionCompletionValues(promptName: promptName)
        case "issue":
            return [
                "click/pop cleanup",
                "reduce muddy low mids",
                "smooth harsh transient",
                "replace weak vocal phrase",
                "remove clipped artifact"
            ]
        case "outputDirectory":
            return await outputDirectoryCompletionValues()
        default:
            return []
        }
    }

    private func resourceCompletionValues(
        resourceURI: String,
        argumentName: String
    ) async -> [String] {
        guard argumentName == "id" || argumentName == "identifier" else {
            return []
        }

        let normalized = resourceURI.lowercased()
        if normalized.contains("model") {
            return await modelCompletionValues()
        }
        if normalized.contains("adapter") {
            return await adapterCompletionValues()
        }
        if normalized.contains("preset") {
            return await presetCompletionValues()
        }
        if normalized.contains("workflow") {
            return MCPPromptCatalog.promptNames
        }
        if normalized.contains("operation") {
            return operationCompletionValues()
        }
        if normalized.contains("trace") {
            return historyCompletionValues(traceOnly: true)
        }
        if normalized.contains("history")
            || normalized.contains("take")
            || normalized.contains("audio") {
            return historyCompletionValues()
        }
        return []
    }

    private func completionResult(
        _ values: [String],
        matching partialValue: String
    ) -> Complete.Result {
        let partial = partialValue.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let uniqueValues = uniqueNonEmpty(values)
        let matches = partial.isEmpty
            ? uniqueValues
            : uniqueValues.filter { $0.lowercased().hasPrefix(partial) }
        let limited = Array(matches.prefix(100))

        return .init(
            completion: .init(
                values: limited,
                total: matches.count,
                hasMore: matches.count > limited.count
            )
        )
    }

    private func uniqueNonEmpty(_ values: [String]) -> [String] {
        var seen = Set<String>()
        var unique: [String] = []
        for value in values {
            let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty, seen.insert(trimmed).inserted else { continue }
            unique.append(trimmed)
        }
        return unique
    }

    private func directionCompletionValues(promptName: String) -> [String] {
        switch promptName {
        case "improve_chorus":
            return [
                "make the chorus more anthemic",
                "tighten the vocal phrasing",
                "add a cleaner synth lift",
                "make it bigger without over-editing"
            ]
        case "create_mix_variants":
            return [
                "brighter and wider",
                "tighter low end",
                "more ambient",
                "vocal-forward"
            ]
        default:
            return [
                "keep it conservative",
                "make it more polished",
                "preserve the original feel"
            ]
        }
    }

    private func historyCompletionValues(traceOnly: Bool = false) -> [String] {
        let payload = facade.listHistory(limit: 100)
        let rows = payload["history"] as? [[String: Any]] ?? []
        var values: [String] = []
        for row in rows {
            if traceOnly, boolArgument(row["hasTrace"]) != true {
                continue
            }
            values.append(contentsOf: stringFields(row, keys: ["id", "fileName", "traceFileName"]))
        }
        return values
    }

    private func modelCompletionValues() async -> [String] {
        let payload = await facade.listModels()
        let rows = payload["models"] as? [[String: Any]] ?? []
        return rows.flatMap { stringFields($0, keys: ["id", "displayName"]) }
    }

    private func adapterCompletionValues() async -> [String] {
        let payload = await facade.listAdapters()
        let rows = payload["adapters"] as? [[String: Any]] ?? []
        return rows.flatMap { stringFields($0, keys: ["id", "name"]) }
    }

    private func presetCompletionValues() async -> [String] {
        guard let payload = try? await facade.readResource(uri: "pimpaudio://presets"),
              let presets = payload["savedGenerationPresets"] as? [[String: Any]] else {
            return []
        }
        return presets.flatMap { stringFields($0, keys: ["id", "name"]) }
    }

    private func operationCompletionValues() -> [String] {
        let payload = facade.listOperations(arguments: ["limit": 50])
        let rows = payload["operations"] as? [[String: Any]] ?? []
        return rows.flatMap { stringFields($0, keys: ["id"]) }
    }

    private func currentRegionCompletionValues(
        key: String,
        defaults: [String]
    ) async -> [String] {
        var values: [String] = []
        if let payload = try? await facade.readResource(uri: "pimpaudio://region/current-selection"),
           let selection = payload["selection"] as? [String: Any],
           let value = decimalString(selection[key]) {
            values.append(value)
        }
        values.append(contentsOf: defaults)
        return values
    }

    private func outputDirectoryCompletionValues() async -> [String] {
        var values = clientRootPathCompletionValues()

        if let payload = try? await facade.readResource(uri: "pimpaudio://clip/current"),
           let path = nonEmptyString(payload["path"]) {
            let directory = URL(fileURLWithPath: path).deletingLastPathComponent()
            values.append(directory.appendingPathComponent("MCPExports").path(percentEncoded: false))
        }

        return values
    }

    private func clientRootPathCompletionValues() -> [String] {
        clientRoots.compactMap { root in
            filePath(fromRootURI: root.uri)
        }
    }

    private func filePath(fromRootURI rootURI: String) -> String? {
        guard let url = URL(string: rootURI), url.isFileURL else {
            return nil
        }
        return url.path(percentEncoded: false)
    }

    private func stringFields(_ payload: [String: Any], keys: [String]) -> [String] {
        keys.compactMap { nonEmptyString(payload[$0]) }
    }

    private func decimalString(_ value: Any?) -> String? {
        guard let number = doubleValue(value) else {
            return nonEmptyString(value)
        }
        if number.rounded() == number {
            return String(Int(number))
        }
        return String(format: "%.3f", number)
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
        if params.uri == MCPResourceCatalog.clientRootsURI {
            let payload = clientRootsResourcePayload()
            return .init(contents: [
                .text(
                    MCPJSON.prettyJSON(payload),
                    uri: params.uri,
                    mimeType: "application/json"
                )
            ])
        }

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

    private func clientRootsResourcePayload() -> [String: Any] {
        [
            "kind": "clientRoots",
            "supported": clientSupportsRoots,
            "listChangedNotifications": clientRootsListChanged,
            "refreshInFlight": isRefreshingClientRoots,
            "refreshedAt": clientRootsRefreshedAt.map { ISO8601DateFormatter().string(from: $0) } ?? NSNull(),
            "lastError": clientRootsLastError ?? NSNull(),
            "count": clientRoots.count,
            "roots": clientRoots.map(clientRootPayload),
            "policy": [
                "advisory": true,
                "historyAudioPreferred": true,
                "defaultRenderLocation": "MCPExports beside the loaded source audio when no output path is provided.",
                "note": "MCP client roots describe client-approved filesystem boundaries. They do not bypass macOS sandboxing or grant PiMP Audio Studio new file permissions."
            ]
        ]
    }

    private func clientRootPayload(_ root: Root) -> [String: Any] {
        var payload: [String: Any] = [
            "uri": root.uri,
            "isFileURL": filePath(fromRootURI: root.uri) != nil
        ]
        if let name = root.name {
            payload["name"] = name
        }
        if let path = filePath(fromRootURI: root.uri) {
            payload["path"] = path
        }
        return payload
    }

    private func executeTool(name: String, arguments: [String: Any]) async throws -> Any {
        switch await confirmedArgumentsIfNeeded(toolName: name, arguments: arguments) {
        case .proceed(let confirmedArguments):
            return try await executeConfirmedTool(name: name, arguments: confirmedArguments)
        case .declined(let payload):
            return payload
        }
    }

    private func executeConfirmedTool(name: String, arguments: [String: Any]) async throws -> Any {
        switch name {
        case "list_models":
            return await facade.listModels()
        case "list_adapters":
            return await facade.listAdapters()
        case "training_status":
            return facade.trainingStatus()
        case "list_training_datasets":
            return facade.listTrainingDatasets(arguments: arguments)
        case "list_training_runs":
            return facade.listTrainingRuns(arguments: arguments)
        case "inspect_training_dataset":
            return try facade.inspectTrainingDataset(arguments: arguments)
        case "inspect_training_run":
            return try facade.inspectTrainingRun(arguments: arguments)
        case "compare_training_runs":
            return try facade.compareTrainingRuns(arguments: arguments)
        case "tail_training_log":
            return try facade.tailTrainingLog(arguments: arguments)
        case "training_preflight":
            return try await facade.trainingPreflight(arguments: arguments)
        case "start_training":
            return try await facade.startTraining(arguments: arguments)
        case "stop_training":
            return try facade.stopTraining(arguments: arguments)
        case "runpod_status":
            return try await facade.runPodStatus(arguments: arguments)
        case "refresh_runpod_launch_options":
            return try await facade.refreshRunPodLaunchOptions(arguments: arguments)
        case "launch_runpod_pod":
            return try await facade.launchRunPodPod(arguments: arguments)
        case "stop_runpod_pod":
            return try await facade.stopRunPodPod(arguments: arguments)
        case "create_remote_training_job":
            return try await facade.createRemoteTrainingJob(arguments: arguments)
        case "upload_remote_training_job":
            return try await facade.uploadRemoteTrainingJob(arguments: arguments)
        case "start_remote_training_job":
            return try await facade.startRemoteTrainingJob(arguments: arguments)
        case "remote_training_job_status":
            return try await facade.remoteTrainingJobStatus(arguments: arguments)
        case "pull_remote_training_job":
            return try await facade.pullRemoteTrainingJob(arguments: arguments)
        case "build_training_dataset":
            return try await facade.buildTrainingDataset(arguments: arguments)
        case "enrich_training_dataset":
            return try await facade.enrichTrainingDataset(arguments: arguments)
        case "update_training_dataset_review":
            return try facade.updateTrainingDatasetReview(arguments: arguments)
        case "import_adapter":
            return try await facade.importAdapter(arguments: arguments)
        case "delete_adapter":
            return try facade.deleteAdapter(arguments: arguments)
        case "merge_adapters":
            return try facade.mergeAdapters(arguments: arguments)
        case "download_model":
            return try await facade.downloadModel(arguments: arguments)
        case "set_active_model":
            return try await facade.setActiveModel(arguments: arguments)
        case "delete_history_entries":
            return try facade.deleteHistoryEntries(arguments: arguments)
        case "save_generation_preset":
            return try facade.saveGenerationPreset(arguments: arguments)
        case "delete_generation_preset":
            return try facade.deleteGenerationPreset(arguments: arguments)
        case "set_verbose_diagnostics":
            return try facade.setVerboseDiagnostics(arguments: arguments)
        case "tail_diagnostics_log":
            return try facade.tailDiagnosticsLog(arguments: arguments)
        case "cancel_generation":
            return try facade.cancelGeneration(arguments: arguments)
        case "run_adapter_audition":
            return try await facade.runAdapterAudition(arguments: arguments)
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
            return try await facade.generate(arguments: arguments)
        case "generation_status":
            return try facade.generationStatus()
        case "editor_status":
            return try await facade.editorStatus()
        case "load_editor_audio":
            return try facade.loadEditorAudio(arguments: arguments)
        case "submit_editor_inpaint":
            return try await facade.submitEditorInpaint(arguments: arguments)
        case "apply_editor_effect_chain":
            return try await facade.applyEditorEffectChain(arguments: arguments)
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
            throw MCPError.methodNotFound("Unknown PiMP Audio Studio tool: \(name)")
        }
    }

    private enum ConfirmationDecision {
        case proceed([String: Any])
        case declined([String: Any])
    }

    private func confirmedArgumentsIfNeeded(
        toolName: String,
        arguments: [String: Any]
    ) async -> ConfirmationDecision {
        switch MCPConfirmationDecision.evaluate(
            requiresConfirmation: PiMPAudioMCPServer.toolRequiresElicitableConfirmation(toolName),
            confirmArgument: boolArgument(arguments["confirm"]),
            isDryRun: isDryRun(toolName: toolName, arguments: arguments),
            clientSupportsFormElicitation: clientSupportsFormElicitation
        ) {
        case .proceed:
            return .proceed(arguments)
        case .declined(let action, let reason):
            return .declined(confirmationDeclinedPayload(
                toolName: toolName,
                action: action,
                reason: reason
            ))
        case .elicit:
            break
        }

        do {
            let result = try await server.requestElicitation(
                message: confirmationMessage(for: toolName),
                requestedSchema: confirmationSchema(for: toolName),
                mode: .form,
                _meta: Metadata(additionalFields: [
                    "pimpAudioTool": .string(toolName),
                    "confirmArgument": .string("confirm")
                ])
            )

            guard result.action == .accept else {
                return .declined(confirmationDeclinedPayload(
                    toolName: toolName,
                    action: result.action.rawValue,
                    reason: "User \(pastTense(result.action)) the MCP confirmation request."
                ))
            }

            guard result.content?["approved"]?.boolValue == true else {
                return .declined(confirmationDeclinedPayload(
                    toolName: toolName,
                    action: result.action.rawValue,
                    reason: "The MCP confirmation response did not set approved=true."
                ))
            }

            var confirmedArguments = arguments
            confirmedArguments["confirm"] = true
            return .proceed(confirmedArguments)
        } catch {
            return .declined(confirmationDeclinedPayload(
                toolName: toolName,
                action: "error",
                reason: "MCP confirmation request failed: \(error.localizedDescription)"
            ))
        }
    }

    private func isDryRun(toolName: String, arguments: [String: Any]) -> Bool {
        let dryRunDefaultTools: Set<String> = [
            "generate",
            "generate_song",
            "load_editor_audio",
            "render_editor_audio",
            "start_training",
            "launch_runpod_pod",
            "create_remote_training_job",
            "build_training_dataset",
            "enrich_training_dataset",
            "download_model"
        ]
        if dryRunDefaultTools.contains(toolName),
           arguments["dryRun"] == nil,
           arguments["validateOnly"] == nil {
            return true
        }
        return boolArgument(arguments["dryRun"] ?? arguments["validateOnly"]) == true
    }

    private func boolArgument(_ value: Any?) -> Bool? {
        switch value {
        case let value as Bool:
            return value
        case let value as NSNumber:
            return value.boolValue
        case let value as String:
            switch value.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
            case "true", "yes", "1":
                return true
            case "false", "no", "0":
                return false
            default:
                return nil
            }
        default:
            return nil
        }
    }

    private func confirmationSchema(for toolName: String) -> Elicitation.RequestSchema {
        Elicitation.RequestSchema(
            title: "Confirm PiMP Audio Studio MCP Action",
            description: "Approve only if this local PiMP Audio Studio action was intentional.",
            properties: [
                "approved": [
                    "type": "boolean",
                    "title": "Approve",
                    "description": "Set true to run \(toolName). Leave false to decline.",
                    "default": false
                ]
            ],
            required: ["approved"]
        )
    }

    private func confirmationMessage(for toolName: String) -> String {
        switch toolName {
        case "generate":
            return "PiMP Audio Studio wants to start a local audio generation. Approve this action?"
        case "generate_song":
            return "PiMP Audio Studio wants to orchestrate a song generation (optional lyrics/adapters, then generate). Approve this action?"
        case "write_lyrics":
            return "PiMP Audio Studio wants to call the lyric writer (may use a paid API). Approve this action?"
        case "apply_adapter":
            return "PiMP Audio Studio wants to change the adapter selection for the next generation. Approve this action?"
        case "load_editor_audio":
            return "PiMP Audio Studio wants to load audio into the editor. Approve this action?"
        case "render_editor_audio":
            return "PiMP Audio Studio wants to export editor audio to disk. Approve this action?"
        case "mark_preferred_take":
            return "PiMP Audio Studio wants to mark a history take as preferred. Approve this action?"
        case "cancel_generation":
            return "PiMP Audio Studio wants to cancel the active generation. Approve this action?"
        case "cancel_operation":
            return "PiMP Audio Studio wants to cancel a running operation. Approve this action?"
        case "submit_editor_inpaint":
            return "PiMP Audio Studio wants to submit a selected-region editor inpaint/repaint mutation. Approve this action?"
        case "apply_editor_effect_chain":
            return "PiMP Audio Studio wants to apply DSP effects to the loaded editor audio. Approve this mutation?"
        case "start_training":
            return "PiMP Audio Studio wants to start a local LoRA/LoKR training run. Approve this action?"
        case "stop_training":
            return "PiMP Audio Studio wants to stop the active local training run. Approve this action?"
        case "launch_runpod_pod":
            return "PiMP Audio Studio wants to launch a billable RunPod GPU pod. Approve this action?"
        case "stop_runpod_pod":
            return "PiMP Audio Studio wants to terminate/delete a RunPod GPU pod. Approve this action?"
        case "create_remote_training_job":
            return "PiMP Audio Studio wants to export a remote training job zip. Approve this action?"
        case "upload_remote_training_job":
            return "PiMP Audio Studio wants to upload a remote job and worker to a RunPod pod. Approve this action?"
        case "start_remote_training_job":
            return "PiMP Audio Studio wants to start remote worker setup/training on a RunPod pod. Approve this action?"
        case "pull_remote_training_job":
            return "PiMP Audio Studio wants to download remote training artifacts and import the adapter. Approve this action?"
        case "build_training_dataset":
            return "PiMP Audio Studio wants to build a local training dataset package. Approve this action?"
        case "enrich_training_dataset":
            return "PiMP Audio Studio wants to enrich a training dataset. Approve this action?"
        case "update_training_dataset_review":
            return "PiMP Audio Studio wants to update dataset review metadata. Approve this action?"
        case "import_adapter":
            return "PiMP Audio Studio wants to import an adapter into the local registry. Approve this action?"
        case "delete_adapter":
            return "PiMP Audio Studio wants to delete an installed adapter. Approve this action?"
        case "merge_adapters":
            return "PiMP Audio Studio wants to merge two LoRA adapters. Approve this action?"
        case "download_model":
            return "PiMP Audio Studio wants to download a model. Approve this action?"
        case "set_active_model":
            return "PiMP Audio Studio wants to change the active model. Approve this action?"
        case "delete_history_entries":
            return "PiMP Audio Studio wants to delete history entries. Approve this action?"
        case "save_generation_preset":
            return "PiMP Audio Studio wants to save a generation preset. Approve this action?"
        case "delete_generation_preset":
            return "PiMP Audio Studio wants to delete a generation preset. Approve this action?"
        case "run_adapter_audition":
            return "PiMP Audio Studio wants to run fixed-seed adapter audition snapshots. Approve this action?"
        case "release_memory":
            return "PiMP Audio Studio wants to cancel active generation if needed, unload model weights, and flush MLX GPU cache. Approve this action?"
        default:
            return "PiMP Audio Studio wants to run \(toolName). Approve this action?"
        }
    }

    private func pastTense(_ action: CreateElicitation.Result.Action) -> String {
        switch action {
        case .accept:
            return "accepted"
        case .decline:
            return "declined"
        case .cancel:
            return "cancelled"
        }
    }

    private func confirmationDeclinedPayload(
        toolName: String,
        action: String,
        reason: String
    ) -> [String: Any] {
        [
            "submitted": false,
            "dryRun": false,
            "tool": toolName,
            "operationID": NSNull(),
            "operation": NSNull(),
            "confirmation": [
                "required": true,
                "accepted": false,
                "mode": "elicitation/form",
                "action": action,
                "reason": reason,
                "confirmArgument": "confirm",
                "fallback": "Clients without MCP elicitation can call again with confirm: true after explicit user approval."
            ]
        ]
    }

    private func canonicalToolName(_ name: String) -> String {
        if name.hasPrefix("pimpaudio.") {
            return String(name.dropFirst("pimpaudio.".count))
        }
        if name.hasPrefix("pimpaudio_") {
            return String(name.dropFirst("pimpaudio_".count))
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

    private func doubleValue(_ value: Any?) -> Double? {
        switch value {
        case let value as Double:
            return value
        case let value as Float:
            return Double(value)
        case let value as Int:
            return Double(value)
        case let value as NSNumber:
            return value.doubleValue
        case let value as String:
            return Double(value)
        default:
            return nil
        }
    }
}

enum MCPValueConverter {
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

private enum MCPPromptCatalog {
    static var promptNames: [String] {
        prompts.map(\.name)
    }

    static let prompts: [Prompt] = [
        prompt(
            name: "improve_chorus",
            title: "Improve Chorus",
            description: "Plan and safely execute a chorus-focused rewrite or polish pass.",
            arguments: [
                argument("takeID", "Optional take/history id to load before editing."),
                argument("regionStart", "Optional chorus region start time in seconds."),
                argument("regionEnd", "Optional chorus region end time in seconds."),
                argument("direction", "Creative direction for the improved chorus.")
            ]
        ),
        prompt(
            name: "create_mix_variants",
            title: "Create Mix Variants",
            description: "Prepare alternate rendered mix treatments from the current take or editor buffer.",
            arguments: [
                argument("takeID", "Optional take/history id to load before creating variants."),
                argument("variantCount", "Requested number of variants. Defaults to three."),
                argument("direction", "Mix direction such as brighter, wider, tighter, or more ambient.")
            ]
        ),
        prompt(
            name: "clean_up_region",
            title: "Clean Up Region",
            description: "Clean or replace a bounded region while preserving surrounding audio.",
            arguments: [
                argument("takeID", "Optional take/history id to load before editing."),
                argument("regionStart", "Region start time in seconds.", required: true),
                argument("regionEnd", "Region end time in seconds.", required: true),
                argument("issue", "Problem to solve, such as click, weak vocal, or muddy transient.")
            ]
        ),
        prompt(
            name: "prepare_stems_for_review",
            title: "Prepare Stems for Review",
            description: "Render review-ready audio files from the current workstation state.",
            arguments: [
                argument("takeID", "Optional take/history id to load before rendering."),
                argument("outputDirectory", "Optional directory for rendered review exports.")
            ]
        ),
        prompt(
            name: "production_ready_prompt",
            title: "Production Ready Prompt",
            description: "Turn a rough music or SFX idea into an PiMP Audio Studio generation request.",
            arguments: [
                argument("prompt", "Rough prompt or creative brief to develop.", required: true),
                argument("duration", "Optional target duration in seconds.")
            ]
        )
    ]

    static func getPrompt(_ params: GetPrompt.Parameters) throws -> GetPrompt.Result {
        switch params.name {
        case "improve_chorus":
            return result(
                description: "Improve a chorus using current PiMP Audio Studio workstation resources.",
                text: """
                Improve the chorus while preserving the best material around it.

                Inputs:
                - takeID: \(argumentValue(params, "takeID", fallback: "use the current loaded clip"))
                - regionStart: \(argumentValue(params, "regionStart", fallback: "infer from pimpaudio://region/current-selection"))
                - regionEnd: \(argumentValue(params, "regionEnd", fallback: "infer from pimpaudio://region/current-selection"))
                - direction: \(argumentValue(params, "direction", fallback: "make the chorus stronger without over-editing"))

                Inspect pimpaudio://takes, pimpaudio://track/current, pimpaudio://clip/current, pimpaudio://region/current-selection, and pimpaudio://presets first.
                If takeID is provided, call load_editor_audio and poll get_operation until complete.
                Validate submit_editor_inpaint and apply_editor_effect_chain with dryRun before any mutation.
                Require confirm: true or MCP elicitation approval before inpaint or DSP changes, then render_editor_audio if the user wants an export.
                """
            )
        case "create_mix_variants":
            return result(
                description: "Create alternate mix renders with bounded DSP chains.",
                text: """
                Create mix variants from the loaded editor audio or takeID \(argumentValue(params, "takeID", fallback: "current clip")).

                Target variant count: \(argumentValue(params, "variantCount", fallback: "3"))
                Direction: \(argumentValue(params, "direction", fallback: "balanced review variants"))

                Inspect pimpaudio://takes, pimpaudio://track/current, pimpaudio://clip/current, and pimpaudio://presets.
                Prefer reversible, bounded DSP chains with apply_editor_effect_chain dryRun first.
                Use render_editor_audio for each approved variant and poll get_operation for completion.
                Do not mark_preferred_take unless the user explicitly asks.
                """
            )
        case "clean_up_region":
            return result(
                description: "Clean or replace a specific editor region safely.",
                text: """
                Clean the specified region only.

                Inputs:
                - takeID: \(argumentValue(params, "takeID", fallback: "current loaded clip"))
                - regionStart: \(argumentValue(params, "regionStart", fallback: "required if no current region exists"))
                - regionEnd: \(argumentValue(params, "regionEnd", fallback: "required if no current region exists"))
                - issue: \(argumentValue(params, "issue", fallback: "remove obvious defect while preserving musical context"))

                Inspect pimpaudio://track/current, pimpaudio://clip/current, pimpaudio://region/current-selection, and pimpaudio://presets.
                Use apply_editor_effect_chain for mechanical cleanup and submit_editor_inpaint for musical replacement.
                Every editor mutation must be scoped to the selected region, dry-run first, and then require confirm: true or MCP elicitation approval.
                """
            )
        case "prepare_stems_for_review":
            return result(
                description: "Render lossless review files from the current workstation state.",
                text: """
                Prepare review-ready renders from the current clip or takeID \(argumentValue(params, "takeID", fallback: "current loaded clip")).

                Output directory: \(argumentValue(params, "outputDirectory", fallback: "default MCPExports folder beside the source audio"))

                Inspect pimpaudio://takes, pimpaudio://track/current, pimpaudio://clip/current, and pimpaudio://operations.
                Use render_editor_audio with lossless output and descriptive filenames.
                Poll get_operation until render operations complete.
                Do not overwrite existing files unless overwrite is explicitly requested.
                """
            )
        case "production_ready_prompt":
            return result(
                description: "Develop a rough idea into a validated PiMP Audio Studio generation request.",
                text: """
                Convert this rough idea into a production-ready PiMP Audio Studio generation request:
                \(argumentValue(params, "prompt", fallback: "No prompt supplied. Ask for a prompt before submitting generation."))

                Target duration: \(argumentValue(params, "duration", fallback: "choose based on prompt and current presets"))

                Inspect pimpaudio://models, pimpaudio://adapters, pimpaudio://presets, and pimpaudio://operations first.
                Use enhance_prompt and write_lyrics when useful.
                Validate the final generate_song call with dryRun before submitting a real generation.
                """
            )
        default:
            throw MCPError.invalidParams("Unknown PiMP Audio Studio prompt: \(params.name)")
        }
    }

    private static func prompt(
        name: String,
        title: String,
        description: String,
        arguments: [Prompt.Argument]
    ) -> Prompt {
        Prompt(
            name: name,
            title: title,
            description: description,
            arguments: arguments
        )
    }

    private static func argument(
        _ name: String,
        _ description: String,
        required: Bool = false
    ) -> Prompt.Argument {
        Prompt.Argument(
            name: name,
            description: description,
            required: required
        )
    }

    private static func argumentValue(
        _ params: GetPrompt.Parameters,
        _ key: String,
        fallback: String
    ) -> String {
        let value = params.arguments?[key]?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        return value.isEmpty ? fallback : value
    }

    private static func result(description: String, text: String) -> GetPrompt.Result {
        .init(
            description: description,
            messages: [.user(.text(text: text))]
        )
    }
}

private enum MCPResourceCatalog {
    static let clientRootsURI = "pimpaudio://client-roots"
    static let clientRootsResourceDefinition: [String: Any] = [
        "uri": clientRootsURI,
        "name": "client-roots",
        "title": "MCP Client Roots",
        "description": "Client-advertised filesystem roots and PiMP Audio Studio local path access policy.",
        "mimeType": "application/json"
    ]

    static let templates: [Resource.Template] = [
        template(
            uriTemplate: "pimpaudio://model/{id}",
            name: "model",
            title: "PiMP Audio Studio Model",
            description: "Model registry entry by model id."
        ),
        template(
            uriTemplate: "pimpaudio://adapter/{id}",
            name: "adapter",
            title: "PiMP Audio Studio Adapter",
            description: "LoRA adapter registry entry by adapter id or name."
        ),
        template(
            uriTemplate: "pimpaudio://training-dataset/{id}",
            name: "training-dataset",
            title: "Training Dataset Detail",
            description: "Detailed local dataset package inspection by id, folder name, or latest."
        ),
        template(
            uriTemplate: "pimpaudio://training-run/{id}",
            name: "training-run",
            title: "Training Run Detail",
            description: "Detailed adapter training run inspection by adapter id, name, or latest."
        ),
        template(
            uriTemplate: "pimpaudio://history/{id}",
            name: "history",
            title: "Generation History Entry",
            description: "Generation history entry by UUID, trace filename, or audio filename."
        ),
        template(
            uriTemplate: "pimpaudio://take/{id}",
            name: "take",
            title: "Generated Take",
            description: "Workstation take view by history id or audio filename."
        ),
        template(
            uriTemplate: "pimpaudio://trace/{id}",
            name: "trace",
            title: "Generation Trace",
            description: "Generation trace sidecar by history id, trace filename, or audio filename."
        ),
        template(
            uriTemplate: "pimpaudio://audio/{id}",
            name: "audio",
            title: "Generated Audio Metadata",
            description: "Generated audio metadata by history id or audio filename."
        ),
        template(
            uriTemplate: "pimpaudio://preset/{id}",
            name: "preset",
            title: "Generation Preset",
            description: "Saved generation preset by UUID or preset name."
        ),
        template(
            uriTemplate: "pimpaudio://workflow/{id}",
            name: "workflow",
            title: "Workflow Prompt",
            description: "Workflow prompt definition by workflow id."
        ),
        template(
            uriTemplate: "pimpaudio://operation/{id}",
            name: "operation",
            title: "MCP Operation",
            description: "Recent MCP-observable operation by operation id."
        )
    ]

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

    private static func template(
        uriTemplate: String,
        name: String,
        title: String,
        description: String
    ) -> Resource.Template {
        Resource.Template(
            uriTemplate: uriTemplate,
            name: name,
            title: title,
            description: description,
            mimeType: "application/json"
        )
    }
}
