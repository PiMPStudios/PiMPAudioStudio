import Darwin
import AppKit
import Foundation

@MainActor
final class ProviderAccessStore: ObservableObject {
    @Published private(set) var statuses: [String: ProviderCLIStatus] = [:]
    @Published private(set) var isRefreshing = false

    private let definitions: [ProviderCLIDefinition]

    init(
        definitions: [ProviderCLIDefinition] = ProviderCLIDefinition.all,
        statuses: [String: ProviderCLIStatus] = [:]
    ) {
        self.definitions = definitions
        self.statuses = statuses
    }

    func status(for providerID: String, displayName: String) -> ProviderCLIStatus {
        statuses[providerID] ?? ProviderCLIStatus.notChecked(
            providerID: providerID,
            displayName: displayName
        )
    }

    func refreshIfNeeded() async {
        guard statuses.isEmpty else { return }
        await refresh()
    }

    func refresh() async {
        guard !isRefreshing else { return }
        isRefreshing = true
        defer { isRefreshing = false }

        var nextStatuses = statuses
        for definition in definitions {
            let status = await ProviderCLIDetector.detect(definition: definition)
            nextStatuses[definition.providerID] = status
            statuses = nextStatuses
        }
    }

    func test(providerID: String) async {
        guard let definition = definitions.first(where: { $0.providerID == providerID }) else { return }

        var current = statuses[providerID] ?? ProviderCLIStatus.notChecked(
            providerID: definition.providerID,
            displayName: definition.displayName
        )
        current.testState = .running
        statuses[providerID] = current

        let detected: ProviderCLIStatus
        if current.command == nil || current.state == .notChecked {
            detected = await ProviderCLIDetector.detect(definition: definition)
            var running = detected
            running.testState = .running
            statuses[providerID] = running
        } else {
            detected = current
        }

        guard let command = detected.command else {
            var missing = statuses[providerID] ?? detected
            missing.testState = .failed("CLI not found")
            statuses[providerID] = missing
            return
        }

        if case .blocked(let detail) = detected.state {
            var blocked = statuses[providerID] ?? detected
            blocked.testState = .blocked(detail)
            statuses[providerID] = blocked
            return
        }

        let outcome = await ProviderCLIDetector.test(definition: definition, command: command)
        var updated = statuses[providerID] ?? detected
        if outcome.isBlocked {
            updated.state = .blocked(outcome.detail)
            updated.detail = outcome.detail
            updated.testState = .blocked(outcome.detail)
        } else if outcome.isWorkspaceTrustNeeded {
            updated.state = .workspaceTrustNeeded(outcome.detail)
            updated.detail = outcome.detail
            updated.testState = .workspaceTrustNeeded(outcome.detail)
        } else {
            updated.testState = outcome.success ? .passed(outcome.detail) : .failed(outcome.detail)
        }
        statuses[providerID] = updated
    }

    func grantAccess(providerID: String) async {
        guard let definition = definitions.first(where: { $0.providerID == providerID }) else { return }
        guard let directoryURL = ProviderCLIAccessGrantPicker.chooseDirectory(for: definition) else { return }

        do {
            try ProviderCLIAccessGrants.saveDirectory(directoryURL, providerID: providerID)
            var current = statuses[providerID] ?? ProviderCLIStatus.notChecked(
                providerID: definition.providerID,
                displayName: definition.displayName
            )
            current.state = .installed
            current.detail = "CLI folder access granted; rechecking"
            current.testState = .idle
            statuses[providerID] = current

            let refreshed = await ProviderCLIDetector.detect(definition: definition)
            statuses[providerID] = refreshed
        } catch {
            var current = statuses[providerID] ?? ProviderCLIStatus.notChecked(
                providerID: definition.providerID,
                displayName: definition.displayName
            )
            current.state = .failed(error.localizedDescription)
            current.detail = error.localizedDescription
            current.testState = .idle
            statuses[providerID] = current
        }
    }
}

struct ProviderCLIStatus: Identifiable, Equatable {
    var id: String { providerID }

    let providerID: String
    let displayName: String
    let command: String?
    let version: String?
    var state: ProviderCLIState
    var detail: String?
    var lastChecked: Date?
    var testState: ProviderCLITestState = .idle

    var canTest: Bool {
        command != nil && state != .missing && !state.isBlocked
    }

    var canGrantAccess: Bool {
        providerID == "xai" && (state.isBlocked || testState.isBlocked)
    }

    static func notChecked(providerID: String, displayName: String) -> ProviderCLIStatus {
        ProviderCLIStatus(
            providerID: providerID,
            displayName: displayName,
            command: nil,
            version: nil,
            state: .notChecked,
            detail: nil,
            lastChecked: nil
        )
    }
}

enum ProviderCLIState: Equatable {
    case notChecked
    case missing
    case installed
    case authenticated
    case unauthenticated
    case workspaceTrustNeeded(String)
    case blocked(String)
    case failed(String)

    var isBlocked: Bool {
        if case .blocked = self {
            return true
        }
        return false
    }
}

enum ProviderCLITestState: Equatable {
    case idle
    case running
    case passed(String)
    case workspaceTrustNeeded(String)
    case blocked(String)
    case failed(String)

    var isBlocked: Bool {
        if case .blocked = self {
            return true
        }
        return false
    }
}

struct ProviderCommandResult: Equatable {
    let command: String
    let arguments: [String]
    let exitStatus: Int32
    let stdout: String
    let stderr: String
    let launchDiagnostic: String?

    init(
        command: String,
        arguments: [String],
        exitStatus: Int32,
        stdout: String,
        stderr: String,
        launchDiagnostic: String? = nil
    ) {
        self.command = command
        self.arguments = arguments
        self.exitStatus = exitStatus
        self.stdout = stdout
        self.stderr = stderr
        self.launchDiagnostic = launchDiagnostic
    }

    var combinedOutput: String {
        [stdout, stderr]
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
            .joined(separator: "\n")
    }

    var cleanedOutput: String {
        combinedOutput.providerAccessCleaned()
    }
}

struct ProviderCLIProbe: Equatable {
    let state: ProviderCLIState
    let detail: String
}

struct ProviderCLITestOutcome: Equatable {
    let success: Bool
    let detail: String
    let isBlocked: Bool
    let isWorkspaceTrustNeeded: Bool

    init(
        success: Bool,
        detail: String,
        isBlocked: Bool = false,
        isWorkspaceTrustNeeded: Bool = false
    ) {
        self.success = success
        self.detail = detail
        self.isBlocked = isBlocked
        self.isWorkspaceTrustNeeded = isWorkspaceTrustNeeded
    }
}

struct ProviderCLIDefinition {
    let providerID: String
    let displayName: String
    let commandCandidates: [String]
    let versionArguments: [String]
    let homeDirectoryMode: ProviderCLIHomeDirectoryMode
    let authProbeArguments: [String]?
    let authParser: ((ProviderCommandResult) -> ProviderCLIProbe)?
    let localAuthProbe: (([String: String]) -> ProviderCLIProbe)?
    let testArguments: [String]?
    let testParser: ((ProviderCommandResult) -> ProviderCLITestOutcome)?

    init(
        providerID: String,
        displayName: String,
        commandCandidates: [String],
        versionArguments: [String],
        homeDirectoryMode: ProviderCLIHomeDirectoryMode = .process,
        authProbeArguments: [String]?,
        authParser: ((ProviderCommandResult) -> ProviderCLIProbe)?,
        localAuthProbe: (([String: String]) -> ProviderCLIProbe)? = nil,
        testArguments: [String]?,
        testParser: ((ProviderCommandResult) -> ProviderCLITestOutcome)?
    ) {
        self.providerID = providerID
        self.displayName = displayName
        self.commandCandidates = commandCandidates
        self.versionArguments = versionArguments
        self.homeDirectoryMode = homeDirectoryMode
        self.authProbeArguments = authProbeArguments
        self.authParser = authParser
        self.localAuthProbe = localAuthProbe
        self.testArguments = testArguments
        self.testParser = testParser
    }

    static let all: [ProviderCLIDefinition] = [
        ProviderCLIDefinition(
            providerID: "anthropic",
            displayName: "Claude Code",
            commandCandidates: ["claude"],
            versionArguments: ["--version"],
            authProbeArguments: ["auth", "status"],
            authParser: ProviderCLIParsers.parseClaudeAuth,
            testArguments: nil,
            testParser: nil
        ),
        ProviderCLIDefinition(
            providerID: "openai",
            displayName: "OpenAI Codex",
            commandCandidates: ["codex"],
            versionArguments: ["--version"],
            homeDirectoryMode: .realUser,
            authProbeArguments: ["login", "status"],
            authParser: ProviderCLIParsers.parseCodexAuth,
            testArguments: nil,
            testParser: nil
        ),
        // Gemini CLI is intentionally not registered for Local CLI Access.
        // Google's free-tier Code Assist / `gemini` client currently rejects lyric
        // runs with IneligibleTierError while OAuth still looks "signed in", which
        // falsely advertises readiness. Gemini remains available via API key.
        ProviderCLIDefinition(
            providerID: "xai",
            displayName: "Grok CLI",
            commandCandidates: ["grok"],
            versionArguments: ["--version"],
            homeDirectoryMode: .realUser,
            authProbeArguments: ["models"],
            authParser: ProviderCLIParsers.parseGrokAuth,
            testArguments: nil,
            testParser: nil
        ),
        ProviderCLIDefinition(
            providerID: "huggingface",
            displayName: "Hugging Face CLI",
            commandCandidates: ["hf", "huggingface-cli"],
            versionArguments: ["--version"],
            homeDirectoryMode: .realUser,
            authProbeArguments: ["auth", "whoami"],
            authParser: ProviderCLIParsers.parseHuggingFaceAuth,
            testArguments: nil,
            testParser: nil
        ),
    ]
}

enum ProviderCLIDetector {
    static func detect(definition: ProviderCLIDefinition) async -> ProviderCLIStatus {
        for command in definition.commandCandidates {
            let versionResult: ProviderCommandResult
            do {
                versionResult = try await ProviderCommandRunner.run(
                    command: command,
                    arguments: definition.versionArguments,
                    homeDirectoryMode: definition.homeDirectoryMode,
                    timeoutSeconds: 5
                )
            } catch {
                if let detail = ProviderCLIParsers.sandboxPermissionDetail(error) {
                    return ProviderCLIStatus(
                        providerID: definition.providerID,
                        displayName: definition.displayName,
                        command: command,
                        version: nil,
                        state: .blocked(detail),
                        detail: detail,
                        lastChecked: Date()
                    )
                }
                return ProviderCLIStatus(
                    providerID: definition.providerID,
                    displayName: definition.displayName,
                    command: command,
                    version: nil,
                    state: .failed(error.localizedDescription),
                    detail: error.localizedDescription,
                    lastChecked: Date()
                )
            }

            guard versionResult.exitStatus == 0 else {
                if ProviderCLIParsers.isCommandMissing(versionResult) {
                    continue
                }
                if let detail = ProviderCLIParsers.sandboxPermissionDetail(versionResult) {
                    return ProviderCLIStatus(
                        providerID: definition.providerID,
                        displayName: definition.displayName,
                        command: command,
                        version: nil,
                        state: .blocked(detail),
                        detail: detail,
                        lastChecked: Date()
                    )
                }
                let detail = ProviderCLIParsers.firstUsefulLine(versionResult.cleanedOutput)
                    ?? "Version probe failed with exit code \(versionResult.exitStatus)"
                return ProviderCLIStatus(
                    providerID: definition.providerID,
                    displayName: definition.displayName,
                    command: command,
                    version: nil,
                    state: .failed(detail),
                    detail: detail,
                    lastChecked: Date()
                )
            }

            let version = ProviderCLIParsers.firstUsefulLine(versionResult.cleanedOutput)
            var state: ProviderCLIState = .installed
            var detail: String? = "Installed"

            if let arguments = definition.authProbeArguments,
               let parser = definition.authParser {
                do {
                    let authResult = try await ProviderCommandRunner.run(
                        command: command,
                        arguments: arguments,
                        homeDirectoryMode: definition.homeDirectoryMode,
                        timeoutSeconds: 10
                    )
                    let probe = parser(authResult)
                    state = probe.state
                    detail = probe.detail
                } catch {
                    if let blockedDetail = ProviderCLIParsers.sandboxPermissionDetail(error) {
                        state = .blocked(blockedDetail)
                        detail = blockedDetail
                    } else {
                        state = .failed(error.localizedDescription)
                        detail = error.localizedDescription
                    }
                }
            } else if let localAuthProbe = definition.localAuthProbe {
                let probe = localAuthProbe(
                    ProviderCommandRunner.commandEnvironment(
                        homeDirectoryMode: definition.homeDirectoryMode
                    )
                )
                state = probe.state
                detail = probe.detail
            } else {
                detail = "Installed; test required for sign-in check"
            }

            return ProviderCLIStatus(
                providerID: definition.providerID,
                displayName: definition.displayName,
                command: command,
                version: version,
                state: state,
                detail: detail,
                lastChecked: Date()
            )
        }

        return ProviderCLIStatus(
            providerID: definition.providerID,
            displayName: definition.displayName,
            command: nil,
            version: nil,
            state: .missing,
            detail: "CLI not found",
            lastChecked: Date()
        )
    }

    static func test(definition: ProviderCLIDefinition, command: String) async -> ProviderCLITestOutcome {
        guard let arguments = definition.testArguments ?? definition.authProbeArguments else {
            return ProviderCLITestOutcome(success: false, detail: "No test command configured")
        }

        do {
            let result = try await ProviderCommandRunner.run(
                command: command,
                arguments: arguments,
                homeDirectoryMode: definition.homeDirectoryMode,
                timeoutSeconds: definition.testArguments == nil ? 10 : 30
            )

            if result.exitStatus != 0,
               let detail = ProviderCLIParsers.sandboxPermissionDetail(result) {
                return ProviderCLITestOutcome(success: false, detail: detail, isBlocked: true)
            }

            if let parser = definition.testParser {
                return parser(result)
            }

            if let authParser = definition.authParser {
                return ProviderCLIParsers.testOutcome(from: authParser(result))
            }

            return ProviderCLIParsers.parseGenericTest(result)
        } catch {
            if let detail = ProviderCLIParsers.sandboxPermissionDetail(error) {
                return ProviderCLITestOutcome(success: false, detail: detail, isBlocked: true)
            }
            return ProviderCLITestOutcome(success: false, detail: error.localizedDescription)
        }
    }
}

enum ProviderCLIParsers {
    static func parseClaudeAuth(_ result: ProviderCommandResult) -> ProviderCLIProbe {
        if let data = result.stdout.data(using: .utf8),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            let loggedIn = json["loggedIn"] as? Bool ?? false
            guard loggedIn else {
                return ProviderCLIProbe(state: .unauthenticated, detail: "Not signed in")
            }

            var parts: [String] = []
            if let method = json["authMethod"] as? String, !method.isEmpty {
                parts.append("via \(method)")
            }
            if let subscription = json["subscriptionType"] as? String, !subscription.isEmpty {
                parts.append(subscription)
            }
            if let email = json["email"] as? String, !email.isEmpty {
                parts.append(email)
            }

            let detail = parts.isEmpty ? "Signed in" : "Signed in " + parts.joined(separator: " - ")
            return ProviderCLIProbe(state: .authenticated, detail: detail)
        }

        guard result.exitStatus == 0 else {
            return failedProbe(result, fallback: "Claude auth probe failed")
        }

        let output = result.cleanedOutput
        if output.localizedCaseInsensitiveContains("logged in") {
            return ProviderCLIProbe(state: .authenticated, detail: firstUsefulLine(output) ?? "Signed in")
        }
        return ProviderCLIProbe(
            state: .installed,
            detail: firstUsefulLine(output) ?? "Installed; auth status unavailable"
        )
    }

    static func parseCodexAuth(_ result: ProviderCommandResult) -> ProviderCLIProbe {
        let output = result.cleanedOutput
        guard result.exitStatus == 0 else {
            if output.localizedCaseInsensitiveContains("not logged in") {
                return ProviderCLIProbe(state: .unauthenticated, detail: "Not signed in")
            }
            return failedProbe(result, fallback: "Codex auth probe failed")
        }

        if output.localizedCaseInsensitiveContains("logged in") {
            return ProviderCLIProbe(
                state: .authenticated,
                detail: codexLoginStatusLine(output) ?? firstUsefulLine(output) ?? "Signed in"
            )
        }
        return ProviderCLIProbe(state: .installed, detail: firstUsefulLine(output) ?? "Installed")
    }

    static func parseGeminiLocalAuth(environment: [String: String]) -> ProviderCLIProbe {
        if let detail = geminiEnvironmentAuthDetail(environment) {
            return ProviderCLIProbe(state: .authenticated, detail: detail)
        }

        let home = ProviderCLIPaths.processHomeDirectory(environment: environment)
        let geminiURL = URL(fileURLWithPath: home, isDirectory: true)
            .appendingPathComponent(".gemini", isDirectory: true)
        let settingsURL = geminiURL.appendingPathComponent("settings.json")

        guard FileManager.default.fileExists(atPath: settingsURL.path) else {
            return ProviderCLIProbe(state: .unauthenticated, detail: "Not signed in")
        }

        do {
            let data = try Data(contentsOf: settingsURL)
            let settings = try JSONSerialization.jsonObject(with: data) as? [String: Any]
            guard let selectedType = geminiSelectedAuthType(from: settings) else {
                return ProviderCLIProbe(state: .unauthenticated, detail: "Not signed in")
            }

            if selectedType.localizedCaseInsensitiveContains("oauth") {
                let oauthURL = geminiURL.appendingPathComponent("oauth_creds.json")
                guard FileManager.default.fileExists(atPath: oauthURL.path) else {
                    return ProviderCLIProbe(state: .unauthenticated, detail: "Google OAuth credentials missing")
                }
                return ProviderCLIProbe(state: .authenticated, detail: "Signed in with Google OAuth")
            }

            return ProviderCLIProbe(state: .authenticated, detail: "Configured with \(selectedType)")
        } catch {
            if let detail = sandboxPermissionDetail(error) {
                return ProviderCLIProbe(state: .blocked(detail), detail: detail)
            }
            return ProviderCLIProbe(
                state: .failed(error.localizedDescription),
                detail: error.localizedDescription
            )
        }
    }

    static func parseGrokAuth(_ result: ProviderCommandResult) -> ProviderCLIProbe {
        let output = result.cleanedOutput
        guard result.exitStatus == 0 else {
            if isGrokUnauthenticated(output) {
                return ProviderCLIProbe(state: .unauthenticated, detail: "Not signed in")
            }
            return failedProbe(result, fallback: "Grok auth probe failed")
        }

        if isGrokUnauthenticated(output) {
            return ProviderCLIProbe(state: .unauthenticated, detail: "Not signed in")
        }

        if output.localizedCaseInsensitiveContains("logged in with grok.com") {
            let defaultModel = output
                .components(separatedBy: .newlines)
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                .first { $0.localizedCaseInsensitiveContains("(default)") }
                .map { $0.replacingOccurrences(of: "(default)", with: "").trimmingCharacters(in: .whitespacesAndNewlines) }

            if let defaultModel, !defaultModel.isEmpty {
                return ProviderCLIProbe(
                    state: .authenticated,
                    detail: "Signed in with grok.com - default \(defaultModel)"
                )
            }
            return ProviderCLIProbe(state: .authenticated, detail: "Signed in with grok.com")
        }

        return ProviderCLIProbe(state: .installed, detail: firstUsefulLine(output) ?? "Installed")
    }

    static func parseHuggingFaceAuth(_ result: ProviderCommandResult) -> ProviderCLIProbe {
        let output = result.cleanedOutput
        guard result.exitStatus == 0 else {
            if output.localizedCaseInsensitiveContains("not logged in") ||
                output.localizedCaseInsensitiveContains("not authenticated") {
                return ProviderCLIProbe(state: .unauthenticated, detail: "Not signed in")
            }
            return failedProbe(result, fallback: "Hugging Face auth probe failed")
        }

        if let userLine = output
            .components(separatedBy: .newlines)
            .map({ $0.trimmingCharacters(in: .whitespacesAndNewlines) })
            .first(where: { $0.hasPrefix("user=") || $0.hasPrefix("user:") }) {
            let user = userLine
                .replacingOccurrences(of: "user=", with: "")
                .replacingOccurrences(of: "user:", with: "")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            return ProviderCLIProbe(state: .authenticated, detail: "Signed in as \(user)")
        }

        return ProviderCLIProbe(state: .authenticated, detail: firstUsefulLine(output) ?? "Signed in")
    }

    static func parseGeminiTest(_ result: ProviderCommandResult) -> ProviderCLITestOutcome {
        guard result.exitStatus == 0 else {
            if let detail = workspaceTrustDetail(result) {
                return ProviderCLITestOutcome(
                    success: false,
                    detail: detail,
                    isWorkspaceTrustNeeded: true
                )
            }
            return ProviderCLITestOutcome(
                success: false,
                detail: firstUsefulLine(result.cleanedOutput) ?? "Gemini test failed"
            )
        }

        if let data = result.stdout.data(using: .utf8),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let response = json["response"] as? String {
            return ProviderCLITestOutcome(
                success: !response.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                detail: "Received response: \(response.trimmingCharacters(in: .whitespacesAndNewlines))"
            )
        }

        let output = result.cleanedOutput
        return ProviderCLITestOutcome(
            success: !output.isEmpty,
            detail: firstUsefulLine(output) ?? "Gemini responded"
        )
    }

    static func parseGenericTest(_ result: ProviderCommandResult) -> ProviderCLITestOutcome {
        if result.exitStatus == 0 {
            return ProviderCLITestOutcome(
                success: true,
                detail: firstUsefulLine(result.cleanedOutput) ?? "Test passed"
            )
        }
        return ProviderCLITestOutcome(
            success: false,
            detail: firstUsefulLine(result.cleanedOutput) ?? "Test failed"
        )
    }

    static func testOutcome(from probe: ProviderCLIProbe) -> ProviderCLITestOutcome {
        switch probe.state {
        case .authenticated:
            return ProviderCLITestOutcome(success: true, detail: probe.detail)
        case .installed:
            return ProviderCLITestOutcome(success: true, detail: probe.detail)
        case .unauthenticated:
            return ProviderCLITestOutcome(success: false, detail: probe.detail)
        case .workspaceTrustNeeded(let message):
            return ProviderCLITestOutcome(
                success: false,
                detail: message,
                isWorkspaceTrustNeeded: true
            )
        case .blocked:
            return ProviderCLITestOutcome(success: false, detail: probe.detail, isBlocked: true)
        case .failed(let message):
            return ProviderCLITestOutcome(success: false, detail: message)
        case .missing:
            return ProviderCLITestOutcome(success: false, detail: "CLI not found")
        case .notChecked:
            return ProviderCLITestOutcome(success: false, detail: "Not checked")
        }
    }

    static func isCommandMissing(_ result: ProviderCommandResult) -> Bool {
        result.exitStatus == 127 ||
            result.cleanedOutput.localizedCaseInsensitiveContains("no such file or directory")
    }

    static func sandboxPermissionDetail(_ result: ProviderCommandResult) -> String? {
        guard let detail = sandboxPermissionDetail(from: result.cleanedOutput) else {
            return nil
        }
        guard let launchDiagnostic = result.launchDiagnostic else {
            return detail
        }
        return "\(detail); \(launchDiagnostic)"
    }

    static func sandboxPermissionDetail(_ error: Error) -> String? {
        sandboxPermissionDetail(from: error.localizedDescription)
    }

    static func sandboxPermissionDetail(from output: String) -> String? {
        let lines = output
            .components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }

        guard let line = lines.first(where: isSandboxPermissionLine) else {
            return nil
        }
        return "Sandbox blocked CLI access: \(line)"
    }

    static func workspaceTrustDetail(_ result: ProviderCommandResult) -> String? {
        let lines = result.cleanedOutput
            .components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }

        guard let line = lines.first(where: isWorkspaceTrustLine) else {
            return nil
        }
        return line
    }

    static func firstUsefulLine(_ output: String) -> String? {
        output
            .components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .first { !$0.isEmpty && !$0.hasPrefix("[WARN]") }
    }

    private static func failedProbe(_ result: ProviderCommandResult, fallback: String) -> ProviderCLIProbe {
        if let detail = sandboxPermissionDetail(result) {
            return ProviderCLIProbe(state: .blocked(detail), detail: detail)
        }
        if let detail = workspaceTrustDetail(result) {
            return ProviderCLIProbe(state: .workspaceTrustNeeded(detail), detail: detail)
        }
        let detail = firstUsefulLine(result.cleanedOutput) ?? fallback
        return ProviderCLIProbe(state: .failed(detail), detail: detail)
    }

    private static func isSandboxPermissionLine(_ line: String) -> Bool {
        let lowercased = line.lowercased()
        return lowercased.contains("operation not permitted") ||
            lowercased.contains("eperm")
    }

    private static func isWorkspaceTrustLine(_ line: String) -> Bool {
        let lowercased = line.lowercased()
        return lowercased.contains("current folder is not trusted") ||
            lowercased.contains("workspace") && lowercased.contains("not trusted")
    }

    private static func geminiEnvironmentAuthDetail(_ environment: [String: String]) -> String? {
        if let apiKey = environment["GEMINI_API_KEY"], !apiKey.isEmpty {
            return "Configured with GEMINI_API_KEY"
        }
        if environment["GOOGLE_GENAI_USE_VERTEXAI"] == "true" {
            return "Configured with Vertex AI"
        }
        if environment["GOOGLE_GENAI_USE_GCA"] == "true" {
            return "Configured with Gemini Code Assist"
        }
        return nil
    }

    private static func geminiSelectedAuthType(from settings: [String: Any]?) -> String? {
        let security = settings?["security"] as? [String: Any]
        let auth = security?["auth"] as? [String: Any]
        let selectedType = (auth?["selectedType"] as? String)?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return selectedType?.isEmpty == false ? selectedType : nil
    }

    private static func codexLoginStatusLine(_ output: String) -> String? {
        output
            .components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .first {
                !$0.isEmpty &&
                    $0.localizedCaseInsensitiveContains("logged in")
            }
    }

    private static func isGrokUnauthenticated(_ output: String) -> Bool {
        output.localizedCaseInsensitiveContains("not logged in") ||
            output.localizedCaseInsensitiveContains("not authenticated")
    }
}

enum ProviderCommandRunner {
    private static let envURL = URL(fileURLWithPath: "/usr/bin/env")

    static func commandEnvironment(
        from baseEnvironment: [String: String] = ProcessInfo.processInfo.environment,
        homeDirectoryMode: ProviderCLIHomeDirectoryMode = .process
    ) -> [String: String] {
        var environment = baseEnvironment
        let home = ProviderCLIPaths.homeDirectory(
            for: homeDirectoryMode,
            environment: environment
        )
        let user = environment["USER"] ?? NSUserName()
        environment["PATH"] = ProviderCLIPaths.executableSearchPath(
            environment: environment,
            primaryHomeDirectory: home
        )
        environment["HOME"] = home
        environment["USER"] = user
        environment["LOGNAME"] = environment["LOGNAME"] ?? user
        environment["SHELL"] = environment["SHELL"] ?? "/bin/zsh"
        environment["TMPDIR"] = environment["TMPDIR"] ?? NSTemporaryDirectory()
        environment["TERM"] = "dumb"
        environment["NO_COLOR"] = "1"
        environment["CLICOLOR"] = "0"
        return environment
    }

    static func run(
        command: String,
        arguments: [String],
        standardInput: String? = nil,
        homeDirectoryMode: ProviderCLIHomeDirectoryMode = .process,
        currentDirectoryURL: URL? = nil,
        timeoutSeconds: TimeInterval
    ) async throws -> ProviderCommandResult {
        let scopedAccess = ProviderCLIAccessGrants.startAccessingGrantedDirectories()
        defer { scopedAccess.forEach { $0.stopAccessing() } }

        let environment = commandEnvironment(homeDirectoryMode: homeDirectoryMode)
        let launch = ProviderCommandLaunch.resolve(
            command: command,
            arguments: arguments,
            environment: environment,
            fallbackURL: envURL
        )
        let capture = ProviderCapturedProcess(
            executableURL: launch.executableURL,
            command: command,
            arguments: arguments,
            processArguments: launch.processArguments,
            standardInput: standardInput,
            environment: environment,
            currentDirectoryURL: currentDirectoryURL,
            launchDiagnostic: launch.diagnostic
        )
        return try await capture.run(timeoutSeconds: timeoutSeconds)
    }
}

enum ProviderCLIHomeDirectoryMode: Equatable, Sendable {
    case process
    case realUser
    /// Explicit HOME for one-shot CLI work that must write under a sandbox-writable root
    /// (e.g. Grok lyric sessions cannot write under the real `~/.grok`, which is read-only).
    case custom(String)
}

struct ProviderCommandLaunch {
    let executableURL: URL
    let processArguments: [String]
    let diagnostic: String

    static func resolve(
        command: String,
        arguments: [String],
        environment: [String: String],
        fallbackURL: URL
    ) -> ProviderCommandLaunch {
        if let executableURL = ProviderCLIPaths.resolvedExecutableURL(
            for: command,
            environment: environment
        ) {
            return ProviderCommandLaunch(
                executableURL: executableURL,
                processArguments: arguments,
                diagnostic: "launch=direct \(executableURL.path)"
            )
        }

        return ProviderCommandLaunch(
            executableURL: fallbackURL,
            processArguments: [command] + arguments,
            diagnostic: "launch=env fallback"
        )
    }
}

enum ProviderCLIAccessGrantPicker {
    @MainActor
    static func chooseDirectory(for definition: ProviderCLIDefinition) -> URL? {
        let panel = NSOpenPanel()
        panel.title = "Grant \(definition.displayName) Access"
        panel.message = "Choose the provider CLI install folder."
        panel.prompt = "Grant Access"
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.directoryURL = ProviderCLIPaths.suggestedGrantDirectory(for: definition)

        return panel.runModal() == .OK ? panel.url : nil
    }
}

enum ProviderCLIAccessGrants {
    private static let bookmarkPrefix = AppUserDefaultsKeys.providerCLIAccessBookmarkPrefix
    private static let legacyBookmarkPrefix = AppUserDefaultsKeys.legacyProviderCLIAccessBookmarkPrefix

    static func saveDirectory(_ url: URL, providerID: String) throws {
        let bookmark = try url.bookmarkData(
            options: [.withSecurityScope],
            includingResourceValuesForKeys: nil,
            relativeTo: nil
        )
        UserDefaults.standard.set(bookmark, forKey: bookmarkKey(providerID: providerID))
    }

    static func grantedDirectoryPaths() -> [String] {
        grantedDirectories()
            .flatMap { url in
                [
                    url.path,
                    url.appendingPathComponent("bin", isDirectory: true).path,
                ]
            }
    }

    static func startAccessingGrantedDirectories() -> [ProviderSecurityScopedAccess] {
        grantedDirectories().map { url in
            ProviderSecurityScopedAccess(
                url: url,
                didStartAccessing: url.startAccessingSecurityScopedResource()
            )
        }
    }

    private static func grantedDirectories() -> [URL] {
        UserDefaults.standard
            .dictionaryRepresentation()
            .keys
            .filter { $0.hasPrefix(bookmarkPrefix) || $0.hasPrefix(legacyBookmarkPrefix) }
            .compactMap { providerID(fromBookmarkKey: $0) }
            .compactMap(resolveDirectory(providerID:))
    }

    private static func resolveDirectory(providerID: String) -> URL? {
        guard let data = UserDefaults.standard.data(forKey: bookmarkKey(providerID: providerID)) ??
            UserDefaults.standard.data(forKey: legacyBookmarkKey(providerID: providerID)) else {
            return nil
        }

        do {
            var isStale = false
            let url = try URL(
                resolvingBookmarkData: data,
                options: [.withSecurityScope],
                relativeTo: nil,
                bookmarkDataIsStale: &isStale
            )
            if isStale {
                try saveDirectory(url, providerID: providerID)
            } else if UserDefaults.standard.data(forKey: bookmarkKey(providerID: providerID)) == nil {
                UserDefaults.standard.set(data, forKey: bookmarkKey(providerID: providerID))
            }
            return url
        } catch {
            UserDefaults.standard.removeObject(forKey: bookmarkKey(providerID: providerID))
            return nil
        }
    }

    private static func bookmarkKey(providerID: String) -> String {
        bookmarkPrefix + providerID
    }

    private static func legacyBookmarkKey(providerID: String) -> String {
        legacyBookmarkPrefix + providerID
    }

    private static func providerID(fromBookmarkKey key: String) -> String? {
        if key.hasPrefix(bookmarkPrefix) {
            return String(key.dropFirst(bookmarkPrefix.count))
        }
        if key.hasPrefix(legacyBookmarkPrefix) {
            return String(key.dropFirst(legacyBookmarkPrefix.count))
        }
        return nil
    }
}

struct ProviderSecurityScopedAccess {
    let url: URL
    let didStartAccessing: Bool

    func stopAccessing() {
        if didStartAccessing {
            url.stopAccessingSecurityScopedResource()
        }
    }
}

enum ProviderCLIPaths {
    private static let standardBinaryDirectories = [
        "/opt/homebrew/bin",
        "/opt/homebrew/sbin",
        "/usr/local/bin",
        "/usr/local/sbin",
        "/usr/bin",
        "/bin",
        "/usr/sbin",
        "/sbin",
    ]

    static func processHomeDirectory(environment: [String: String] = ProcessInfo.processInfo.environment) -> String {
        if let home = environment["HOME"]?.trimmingCharacters(in: .whitespacesAndNewlines),
           !home.isEmpty {
            return home
        }
        return NSHomeDirectory()
    }

    static func realUserHomeDirectory(environment: [String: String] = ProcessInfo.processInfo.environment) -> String {
        let environmentHome = environment["HOME"]?.trimmingCharacters(in: .whitespacesAndNewlines)
        let user = environment["USER"]?.trimmingCharacters(in: .whitespacesAndNewlines)
            ?? NSUserName()

        if let environmentHome, !environmentHome.isEmpty, !isSandboxContainerHome(environmentHome) {
            return environmentHome
        }

        if !user.isEmpty,
           let accountHome = NSHomeDirectoryForUser(user),
           !accountHome.isEmpty,
           !isSandboxContainerHome(accountHome) {
            return accountHome
        }

        if !user.isEmpty {
            return "/Users/\(user)"
        }

        if let environmentHome, !environmentHome.isEmpty {
            return environmentHome
        }

        return NSHomeDirectory()
    }

    static func homeDirectory(
        for mode: ProviderCLIHomeDirectoryMode,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> String {
        switch mode {
        case .process:
            return processHomeDirectory(environment: environment)
        case .realUser:
            return realUserHomeDirectory(environment: environment)
        case .custom(let path):
            return path
        }
    }

    static func executableSearchPath(
        environment: [String: String] = ProcessInfo.processInfo.environment,
        primaryHomeDirectory: String? = nil
    ) -> String {
        let processHome = primaryHomeDirectory ?? processHomeDirectory(environment: environment)
        let userHome = realUserHomeDirectory(environment: environment)
        let includeGrantedDirectories =
            environment["PIMP_AUDIO_DISABLE_PROVIDER_CLI_GRANTS"] != "1"
        var directories = includeGrantedDirectories ? ProviderCLIAccessGrants.grantedDirectoryPaths() : []
        directories.append(contentsOf: [
            "\(processHome)/.grok/bin",
            "\(processHome)/.local/bin",
        ])

        directories.append(contentsOf: standardBinaryDirectories)

        if let existingPath = environment["PATH"], !existingPath.isEmpty {
            directories.append(contentsOf: existingPath.split(separator: ":").map(String.init))
        }

        if userHome != processHome {
            directories.append("\(userHome)/.grok/bin")
            directories.append("\(userHome)/.local/bin")
        }

        return unique(directories).joined(separator: ":")
    }

    static func suggestedGrantDirectory(for definition: ProviderCLIDefinition) -> URL? {
        switch definition.providerID {
        case "xai":
            return URL(
                fileURLWithPath: "\(realUserHomeDirectory())/.grok",
                isDirectory: true
            )
        default:
            return nil
        }
    }

    static func resolvedExecutableURL(
        for command: String,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> URL? {
        if command.contains("/") {
            return launchCandidateURL(atPath: command)
        }

        let directories = (environment["PATH"] ?? executableSearchPath(environment: environment))
            .split(separator: ":")
            .map(String.init)

        for directory in directories {
            let path = URL(fileURLWithPath: directory)
                .appendingPathComponent(command)
                .path
            if let url = launchCandidateURL(atPath: path) {
                return url
            }
        }

        return nil
    }

    private static func isSandboxContainerHome(_ path: String) -> Bool {
        path.contains("/Library/Containers/")
    }

    private static func unique(_ values: [String]) -> [String] {
        var seen = Set<String>()
        return values.filter { value in
            seen.insert(value).inserted
        }
    }

    private static func launchCandidateURL(atPath path: String) -> URL? {
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: path, isDirectory: &isDirectory),
              !isDirectory.boolValue else {
            return nil
        }

        let url = URL(fileURLWithPath: path).resolvingSymlinksInPath()
        if FileManager.default.isExecutableFile(atPath: url.path) ||
            FileManager.default.isExecutableFile(atPath: path) {
            return url
        }

        return isKnownProviderInstallPath(path) ? url : nil
    }

    private static func isKnownProviderInstallPath(_ path: String) -> Bool {
        path.contains("/.grok/bin/") || path.contains("/.grok/downloads/")
    }
}

enum ProviderCommandError: LocalizedError {
    case timedOut(String, TimeInterval)
    case launchFailed(String, String, Error)

    var errorDescription: String? {
        switch self {
        case .timedOut(let command, let timeout):
            return "\(command) timed out after \(Int(timeout)) seconds"
        case .launchFailed(let command, let diagnostic, let error):
            return "\(command) launch failed: \(error.localizedDescription); \(diagnostic)"
        }
    }
}

private final class ProviderCapturedProcess: @unchecked Sendable {
    private let process = Process()
    private let stdout = Pipe()
    private let stderr = Pipe()
    private let lock = NSLock()
    private let command: String
    private let arguments: [String]
    private let launchDiagnostic: String
    private let standardInputData: Data?
    private let standardInputPipe: Pipe?
    private var stdoutData = Data()
    private var stderrData = Data()
    private var continuation: CheckedContinuation<ProviderCommandResult, Error>?
    private var didResume = false
    private var didCancel = false

    init(
        executableURL: URL,
        command: String,
        arguments: [String],
        processArguments: [String],
        standardInput: String?,
        environment: [String: String],
        currentDirectoryURL: URL? = nil,
        launchDiagnostic: String
    ) {
        self.command = command
        self.arguments = arguments
        self.launchDiagnostic = launchDiagnostic
        self.standardInputData = standardInput?.data(using: .utf8)
        self.standardInputPipe = standardInput == nil ? nil : Pipe()
        process.executableURL = executableURL
        process.arguments = processArguments
        process.environment = environment
        process.standardOutput = stdout
        process.standardError = stderr
        if let currentDirectoryURL {
            process.currentDirectoryURL = currentDirectoryURL
        }
        if let standardInputPipe {
            process.standardInput = standardInputPipe
        }
    }

    func run(timeoutSeconds: TimeInterval) async throws -> ProviderCommandResult {
        try await withThrowingTaskGroup(of: ProviderCommandResult.self) { group in
            group.addTask {
                try await self.runProcess()
            }
            group.addTask {
                try await Task.sleep(nanoseconds: UInt64(timeoutSeconds * 1_000_000_000))
                self.cancel()
                throw ProviderCommandError.timedOut(self.command, timeoutSeconds)
            }

            guard let result = try await group.next() else {
                throw CancellationError()
            }
            group.cancelAll()
            return result
        }
    }

    private func runProcess() async throws -> ProviderCommandResult {
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<ProviderCommandResult, Error>) in
                start(continuation: continuation)
            }
        } onCancel: {
            cancel()
        }
    }

    private func start(continuation: CheckedContinuation<ProviderCommandResult, Error>) {
        lock.lock()
        if didCancel {
            lock.unlock()
            continuation.resume(throwing: CancellationError())
            return
        }
        self.continuation = continuation
        lock.unlock()

        stdout.fileHandleForReading.readabilityHandler = { [weak self] handle in
            self?.appendStdout(handle.availableData)
        }
        stderr.fileHandleForReading.readabilityHandler = { [weak self] handle in
            self?.appendStderr(handle.availableData)
        }
        process.terminationHandler = { [weak self] process in
            self?.finish(process: process)
        }

        do {
            try process.run()
            writeStandardInput()
        } catch {
            finish(error: ProviderCommandError.launchFailed(command, launchDiagnostic, error))
        }
    }

    private func writeStandardInput() {
        guard let standardInputData, let standardInputPipe else { return }
        let writer = standardInputPipe.fileHandleForWriting
        DispatchQueue.global(qos: .utility).async {
            try? writer.write(contentsOf: standardInputData)
            try? writer.close()
        }
    }

    private func appendStdout(_ data: Data) {
        guard !data.isEmpty else { return }
        lock.lock()
        stdoutData.append(data)
        lock.unlock()
    }

    private func appendStderr(_ data: Data) {
        guard !data.isEmpty else { return }
        lock.lock()
        stderrData.append(data)
        lock.unlock()
    }

    private func finish(process: Process) {
        stdout.fileHandleForReading.readabilityHandler = nil
        stderr.fileHandleForReading.readabilityHandler = nil
        let remainingStdout = stdout.fileHandleForReading.readDataToEndOfFile()
        let remainingStderr = stderr.fileHandleForReading.readDataToEndOfFile()

        lock.lock()
        stdoutData.append(remainingStdout)
        stderrData.append(remainingStderr)
        guard !didResume, let continuation else {
            lock.unlock()
            return
        }
        didResume = true
        self.continuation = nil
        let outData = stdoutData
        let errData = stderrData
        lock.unlock()

        continuation.resume(returning: ProviderCommandResult(
            command: command,
            arguments: arguments,
            exitStatus: process.terminationStatus,
            stdout: String(data: outData, encoding: .utf8) ?? "",
            stderr: String(data: errData, encoding: .utf8) ?? "",
            launchDiagnostic: launchDiagnostic
        ))
    }

    private func finish(error: Error) {
        stdout.fileHandleForReading.readabilityHandler = nil
        stderr.fileHandleForReading.readabilityHandler = nil
        try? standardInputPipe?.fileHandleForWriting.close()

        lock.lock()
        guard !didResume, let continuation else {
            lock.unlock()
            return
        }
        didResume = true
        self.continuation = nil
        lock.unlock()

        continuation.resume(throwing: error)
    }

    private func cancel() {
        stdout.fileHandleForReading.readabilityHandler = nil
        stderr.fileHandleForReading.readabilityHandler = nil
        try? standardInputPipe?.fileHandleForWriting.close()

        lock.lock()
        didCancel = true
        let continuation = didResume ? nil : self.continuation
        if continuation != nil {
            didResume = true
            self.continuation = nil
        }
        lock.unlock()

        if process.isRunning {
            let pid = process.processIdentifier
            process.terminate()
            DispatchQueue.global(qos: .utility).asyncAfter(deadline: .now() + 2.0) { [process] in
                if process.isRunning {
                    kill(pid, SIGKILL)
                }
            }
        }

        continuation?.resume(throwing: CancellationError())
    }
}

private extension String {
    func providerAccessCleaned() -> String {
        replacingOccurrences(
            of: "\u{001B}\\[[0-9;]*[A-Za-z]",
            with: "",
            options: .regularExpression
        )
        .trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
