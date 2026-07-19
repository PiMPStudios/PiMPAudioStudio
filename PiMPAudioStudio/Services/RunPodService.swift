import Foundation
import PiMPAudioCore
import SwiftUI

@MainActor
final class RunPodService: ObservableObject {
    @Published private(set) var pods: [RunPodPodSummary] = []
    @Published private(set) var gpuTypes: [RunPodGPUType] = []
    @Published private(set) var podTemplates: [RunPodTemplate] = []
    @Published private(set) var isRefreshing = false
    @Published private(set) var isRefreshingGPUTypes = false
    @Published private(set) var isLaunching = false
    @Published private(set) var terminatingPodID: String?
    @Published private(set) var remoteJobStates: [String: RunPodRemoteJobState] = [:]
    @Published var lastMessage: String?
    @Published var lastError: String?

    private let apiKeyStore: APIKeyStore
    private let client: RunPodAPIClient
    private let remoteExecutor: any RunPodRemoteCommandExecuting
    private var remoteMonitorTasks: [String: Task<Void, Never>] = [:]

    init(
        apiKeyStore: APIKeyStore,
        client: RunPodAPIClient = RunPodAPIClient(),
        remoteExecutor: any RunPodRemoteCommandExecuting = RunPodRemoteProcessExecutor()
    ) {
        self.apiKeyStore = apiKeyStore
        self.client = client
        self.remoteExecutor = remoteExecutor
    }

    var hasAPIKey: Bool {
        apiKeyStore.hasKey(for: APIKeyStore.runPodProviderID)
    }

    var keySource: APIKeyStore.KeySource {
        apiKeyStore.keySource(for: APIKeyStore.runPodProviderID)
    }

    func saveAPIKey(_ key: String) {
        let trimmed = key.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        apiKeyStore.setAPIKey(trimmed, for: APIKeyStore.runPodProviderID)
        note("apiKeySaved", "source=\(keySourceDiagnosticsLabel)")
        lastMessage = "RunPod API key saved."
        lastError = nil
    }

    func removeAPIKey() {
        apiKeyStore.removeAPIKey(for: APIKeyStore.runPodProviderID)
        pods = []
        gpuTypes = []
        podTemplates = []
        note("apiKeyRemoved", "source=none")
        lastMessage = "RunPod API key removed."
        lastError = nil
    }

    func refreshPods() async {
        guard let token = runPodToken() else {
            lastError = "Add a RunPod API key before refreshing pods."
            return
        }

        isRefreshing = true
        defer { isRefreshing = false }

        do {
            note("pods/refreshBegin", "source=\(keySourceDiagnosticsLabel)")
            pods = try await client.listPods(token: token)
            note(
                "pods/refreshEnd",
                "count=\(pods.count) ids=\(pods.map(\.id).prefix(12).joined(separator: ","))"
            )
            lastMessage = "Loaded \(pods.count) RunPod pod\(pods.count == 1 ? "" : "s")."
            lastError = nil
        } catch {
            note("pods/refreshError", "error=\(RunPodError.userFacingMessage(for: error))")
            lastError = RunPodError.userFacingMessage(for: error)
        }
    }

    func refreshGPUTypes() async {
        guard let token = runPodToken() else {
            lastError = "Add a RunPod API key before refreshing GPU types."
            return
        }

        isRefreshingGPUTypes = true
        defer { isRefreshingGPUTypes = false }

        do {
            note("gpuTypes/refreshBegin", "source=\(keySourceDiagnosticsLabel)")
            gpuTypes = try await client.listGPUTypes(token: token)
            note(
                "gpuTypes/refreshEnd",
                "count=\(gpuTypes.count) preview=\(gpuTypes.map(\.selectionValue).prefix(8).joined(separator: " | "))"
            )
            lastMessage = "Loaded \(gpuTypes.count) RunPod GPU type\(gpuTypes.count == 1 ? "" : "s")."
            lastError = nil
        } catch {
            note("gpuTypes/refreshError", "error=\(RunPodError.userFacingMessage(for: error))")
            lastError = RunPodError.userFacingMessage(for: error)
        }
    }

    func refreshLaunchOptions() async {
        guard let token = runPodToken() else {
            lastError = "Add a RunPod API key before refreshing RunPod launch options."
            return
        }

        isRefreshingGPUTypes = true
        defer { isRefreshingGPUTypes = false }

        do {
            note("launchOptions/refreshBegin", "source=\(keySourceDiagnosticsLabel)")
            let loadedGPUTypes = try await client.listGPUTypes(token: token)
            let loadedTemplates = try await client.listOfficialPodTemplates(token: token)
            gpuTypes = loadedGPUTypes
            podTemplates = loadedTemplates
            let recommended = RunPodTemplate.recommendedTrainingTemplate(
                for: loadedGPUTypes.first?.selectionValue ?? "",
                from: loadedTemplates
            )
            note(
                "launchOptions/refreshEnd",
                "gpuCount=\(loadedGPUTypes.count) templateCount=\(loadedTemplates.count) firstGPU=\(loadedGPUTypes.first?.selectionValue ?? "none") recommendedTemplate=\(recommended?.id ?? "none")"
            )
            lastMessage = "Loaded \(loadedGPUTypes.count) GPU type\(loadedGPUTypes.count == 1 ? "" : "s") and \(loadedTemplates.count) official RunPod template\(loadedTemplates.count == 1 ? "" : "s")."
            lastError = nil
        } catch {
            note("launchOptions/refreshError", "error=\(RunPodError.userFacingMessage(for: error))")
            lastError = RunPodError.userFacingMessage(for: error)
        }
    }

    func recommendedTrainingTemplate(for gpuTypeValue: String) -> RunPodTemplate? {
        RunPodTemplate.recommendedTrainingTemplate(
            for: gpuTypeValue,
            from: podTemplates
        )
    }

    @discardableResult
    func launchPod(configuration: RunPodLaunchConfiguration) async -> RunPodPodSummary? {
        guard let token = runPodToken() else {
            lastError = "Add a RunPod API key before launching a pod."
            return nil
        }

        isLaunching = true
        defer { isLaunching = false }

        do {
            note("launch/begin", configuration.diagnosticsSummary)
            let pod = try await client.createPod(token: token, configuration: configuration)
            upsert(pod)
            note("launch/success", pod.diagnosticsSummary)
            lastMessage = "RunPod launch requested for \(pod.displayName)."
            lastError = nil
            return pod
        } catch {
            note("launch/error", "configuration={\(configuration.diagnosticsSummary)} error=\(RunPodError.userFacingMessage(for: error))")
            lastError = RunPodError.userFacingMessage(for: error)
            return nil
        }
    }

    func terminatePod(id: String) async {
        guard let token = runPodToken() else {
            lastError = "Add a RunPod API key before terminating a pod."
            return
        }

        terminatingPodID = id
        defer { terminatingPodID = nil }

        do {
            note("terminate/begin", "podID=\(id)")
            try await client.deletePod(token: token, podID: id)
            remoteMonitorTasks.removeValue(forKey: id)?.cancel()
            pods.removeAll { $0.id == id }
            note("terminate/success", "podID=\(id)")
            lastMessage = "Terminate requested for RunPod pod \(id)."
            lastError = nil
            await refreshPods()
        } catch {
            note("terminate/error", "podID=\(id) error=\(RunPodError.userFacingMessage(for: error))")
            lastError = RunPodError.userFacingMessage(for: error)
        }
    }

    var stoppingPodID: String? {
        terminatingPodID
    }

    func remoteJobState(
        for pod: RunPodPodSummary,
        archiveURL: URL?
    ) -> RunPodRemoteJobState? {
        guard let archiveURL,
              let state = remoteJobStates[pod.id],
              state.archivePath == archiveURL.path(percentEncoded: false) else {
            return nil
        }
        return state
    }

    func uploadRemoteJob(
        to pod: RunPodPodSummary,
        archiveURL: URL,
        sshKeyPath: String,
        workerDirectoryURL: URL = RunPodRemoteWorkerLocator.defaultDirectoryURL()
    ) async {
        do {
            let plan = try RunPodRemoteJobPlan(
                pod: pod,
                archiveURL: archiveURL,
                sshKeyPath: sshKeyPath,
                workerDirectoryURL: workerDirectoryURL
            )
            updateRemoteJobState(
                podID: pod.id,
                archiveURL: archiveURL,
                phase: .uploading,
                message: "Preparing \(pod.displayName) and uploading the job plus worker."
            )
            try await runRemoteCommand(plan.prepareUploadCommand)
            try await runRemoteCommand(plan.uploadArchiveCommand)
            try await runRemoteCommand(plan.uploadWorkerCommand)
            updateRemoteJobState(
                podID: pod.id,
                archiveURL: archiveURL,
                phase: .uploaded,
                message: "Upload complete. The worker is ready to start."
            )
            lastMessage = "Uploaded \(archiveURL.lastPathComponent) and the PiMP worker to \(pod.displayName)."
            lastError = nil
        } catch is CancellationError {
            updateRemoteJobState(
                podID: pod.id,
                archiveURL: archiveURL,
                phase: .failed,
                message: "Remote upload was cancelled."
            )
        } catch {
            failRemoteJob(podID: pod.id, archiveURL: archiveURL, error: error)
        }
    }

    func startRemoteJob(
        on pod: RunPodPodSummary,
        archiveURL: URL,
        sshKeyPath: String
    ) async {
        do {
            let plan = try RunPodRemoteJobPlan(
                pod: pod,
                archiveURL: archiveURL,
                sshKeyPath: sshKeyPath
            )
            updateRemoteJobState(
                podID: pod.id,
                archiveURL: archiveURL,
                phase: .starting,
                message: "Preparing the remote worker. First setup typically takes 20\u{2013}45 minutes; progress refreshes automatically."
            )
            beginMonitoring(plan: plan)
            let response = try await runRemoteCommand(
                plan.startCommand,
                timeoutSeconds: 120
            )
            let detail = response.stdout
                .trimmingCharacters(in: .whitespacesAndNewlines)
            updateRemoteJobState(
                podID: pod.id,
                archiveURL: archiveURL,
                phase: .starting,
                message: detail.isEmpty
                    ? "Remote setup started. Progress refreshes automatically."
                    : String(detail.suffix(1_000))
            )
            lastMessage = "Remote setup started on \(pod.displayName)."
            lastError = nil
            beginMonitoring(plan: plan)
        } catch is CancellationError {
            updateRemoteJobState(
                podID: pod.id,
                archiveURL: archiveURL,
                phase: .failed,
                message: "Starting the remote worker was cancelled."
            )
        } catch {
            failRemoteJob(podID: pod.id, archiveURL: archiveURL, error: error)
        }
    }

    func refreshRemoteJob(
        on pod: RunPodPodSummary,
        archiveURL: URL,
        sshKeyPath: String
    ) async {
        do {
            let plan = try RunPodRemoteJobPlan(
                pod: pod,
                archiveURL: archiveURL,
                sshKeyPath: sshKeyPath
            )
            try await refreshRemoteJob(plan: plan)
        } catch is CancellationError {
            return
        } catch {
            failRemoteJob(podID: pod.id, archiveURL: archiveURL, error: error)
        }
    }

    func downloadRemoteJobArtifacts(
        from pod: RunPodPodSummary,
        archiveURL: URL,
        sshKeyPath: String
    ) async throws -> RunPodRemoteJobPullback {
        let plan = try RunPodRemoteJobPlan(
            pod: pod,
            archiveURL: archiveURL,
            sshKeyPath: sshKeyPath
        )
        updateRemoteJobState(
            podID: pod.id,
            archiveURL: archiveURL,
            phase: .downloading,
            message: "Preparing and downloading remote artifacts."
        )
        do {
            try await runRemoteCommand(plan.preparePullbackCommand)
            let destination = try plan.makeLocalPullbackDirectory()
            let pullCommand = plan.pullArtifactsCommand(destinationURL: destination)
            try await runRemoteCommand(pullCommand, timeoutSeconds: 1_800)

            let artifactsURL = destination.appendingPathComponent(
                "remote_artifacts",
                isDirectory: true
            )
            let resultURL = artifactsURL.appendingPathComponent("result.json")
            let result = try RemoteTrainingJobCodec.decoder.decode(
                RemoteTrainingJobResult.self,
                from: Data(contentsOf: resultURL)
            )
            guard result.pimpAudioRemoteResultVersion
                    == RemoteTrainingJobResult.schemaVersion else {
                throw RunPodRemoteJobError.invalidResult(
                    "unsupported result schema \(result.pimpAudioRemoteResultVersion)"
                )
            }
            guard result.status == .succeeded else {
                throw RunPodRemoteJobError.invalidResult(
                    result.error?.message ?? "remote worker did not report success"
                )
            }
            let finalAdapterURL = artifactsURL.appendingPathComponent(
                "final_adapter",
                isDirectory: true
            )
            var isDirectory: ObjCBool = false
            guard FileManager.default.fileExists(
                atPath: finalAdapterURL.path(percentEncoded: false),
                isDirectory: &isDirectory
            ), isDirectory.boolValue else {
                throw RunPodRemoteJobError.invalidResult(
                    "final_adapter directory is missing from the pullback"
                )
            }

            let pullback = RunPodRemoteJobPullback(
                directoryURL: destination,
                artifactsURL: artifactsURL,
                finalAdapterURL: finalAdapterURL,
                result: result
            )
            remoteJobStates[pod.id] = RunPodRemoteJobState(
                archivePath: archiveURL.path(percentEncoded: false),
                phase: .downloaded,
                message: "Artifacts downloaded and validated. Importing the final adapter is safe.",
                logTail: remoteJobStates[pod.id]?.logTail ?? "",
                result: result,
                localArtifactsPath: artifactsURL.path(percentEncoded: false),
                importedAdapterName: nil
            )
            lastMessage = "Downloaded validated remote artifacts from \(pod.displayName)."
            lastError = nil
            return pullback
        } catch {
            failRemoteJob(podID: pod.id, archiveURL: archiveURL, error: error)
            throw error
        }
    }

    func markRemoteAdapterImported(
        podID: String,
        archiveURL: URL,
        adapterName: String
    ) {
        guard var state = remoteJobStates[podID],
              state.archivePath == archiveURL.path(percentEncoded: false) else {
            return
        }
        state.phase = .imported
        state.importedAdapterName = adapterName
        state.message = "Imported \(adapterName) into the local adapter registry."
        remoteJobStates[podID] = state
    }

    private func beginMonitoring(plan: RunPodRemoteJobPlan) {
        remoteMonitorTasks.removeValue(forKey: plan.podID)?.cancel()
        remoteMonitorTasks[plan.podID] = Task { [weak self] in
            guard let self else { return }
            while !Task.isCancelled {
                do {
                    try await Task.sleep(for: .seconds(5))
                    try Task.checkCancellation()
                    await self.refreshRemoteJob(plan: plan)
                    guard let state = self.remoteJobStates[plan.podID],
                          !state.phase.isTerminal else {
                        return
                    }
                } catch {
                    return
                }
            }
        }
    }

    private func refreshRemoteJob(plan: RunPodRemoteJobPlan) async {
        do {
            let response = try await runRemoteCommand(plan.statusCommand)
            let snapshot = try RunPodRemoteStatusSnapshot.parse(response.stdout)
            var state = remoteJobStates[plan.podID] ?? RunPodRemoteJobState(
                archivePath: plan.archiveURL.path(percentEncoded: false),
                phase: .running,
                message: "Remote worker is running.",
                logTail: "",
                result: nil,
                localArtifactsPath: nil,
                importedAdapterName: nil
            )
            state.archivePath = plan.archiveURL.path(percentEncoded: false)
            state.logTail = snapshot.logTail
            switch snapshot.status {
            case .running:
                state.phase = .running
                state.message = "Remote worker is running. Status refreshes automatically."
            case .pending:
                state.phase = .starting
                state.message = snapshot.logTail.isEmpty
                    ? "Preparing the remote worker. First setup typically takes 20\u{2013}45 minutes."
                    : String(snapshot.logTail.prefix(1_000))
                state.logTail = ""
            case .failed:
                state.phase = .failed
                state.message = snapshot.logTail.isEmpty
                    ? "The remote worker stopped without writing result.json."
                    : String(snapshot.logTail.suffix(1_000))
            case .result:
                guard let result = snapshot.result else {
                    throw RunPodRemoteJobError.invalidResult(
                        "result marker did not contain result.json"
                    )
                }
                state.result = result
                state.phase = result.status == .succeeded ? .succeeded : .failed
                state.message = result.status == .succeeded
                    ? "Remote training succeeded. Pull back and import the adapter."
                    : (result.error?.message ?? "Remote training failed.")
            }
            remoteJobStates[plan.podID] = state
        } catch is CancellationError {
            return
        } catch {
            failRemoteJob(
                podID: plan.podID,
                archiveURL: plan.archiveURL,
                error: error
            )
        }
    }

    private func runRemoteCommand(
        _ command: RunPodRemoteCommand,
        timeoutSeconds: TimeInterval = 300
    ) async throws -> RunPodRemoteCommandResult {
        let result = try await remoteExecutor.run(
            executableURL: command.executableURL,
            arguments: command.arguments,
            timeoutSeconds: timeoutSeconds
        )
        guard result.exitStatus == 0 else {
            let detail = result.stderr
                .trimmingCharacters(in: .whitespacesAndNewlines)
            throw RunPodRemoteJobError.commandFailed(
                detail.isEmpty
                    ? "\(command.displayName) exited with status \(result.exitStatus)"
                    : String(detail.suffix(2_000))
            )
        }
        return result
    }

    private func updateRemoteJobState(
        podID: String,
        archiveURL: URL,
        phase: RunPodRemoteJobPhase,
        message: String
    ) {
        remoteJobStates[podID] = RunPodRemoteJobState(
            archivePath: archiveURL.path(percentEncoded: false),
            phase: phase,
            message: message,
            logTail: remoteJobStates[podID]?.logTail ?? "",
            result: remoteJobStates[podID]?.result,
            localArtifactsPath: remoteJobStates[podID]?.localArtifactsPath,
            importedAdapterName: remoteJobStates[podID]?.importedAdapterName
        )
    }

    private func failRemoteJob(
        podID: String,
        archiveURL: URL,
        error: Error
    ) {
        remoteMonitorTasks.removeValue(forKey: podID)?.cancel()
        let message = RunPodRemoteJobError.userFacingMessage(for: error)
        updateRemoteJobState(
            podID: podID,
            archiveURL: archiveURL,
            phase: .failed,
            message: message
        )
        lastError = message
    }

    private func runPodToken() -> String? {
        apiKeyStore.runPodToken()
    }

    private func upsert(_ pod: RunPodPodSummary) {
        if let index = pods.firstIndex(where: { $0.id == pod.id }) {
            pods[index] = pod
        } else {
            pods.insert(pod, at: 0)
        }
    }

    private var keySourceDiagnosticsLabel: String {
        switch keySource {
        case .keychain:
            return "keychain"
        case .environment:
            return "environment"
        case .none:
            return "none"
        }
    }

    private func note(_ tag: String, _ message: String) {
        AudioDiagnostics.note("runpod/\(tag)", message)
    }
}

enum RunPodRemoteJobPhase: String, Equatable, Sendable {
    case uploading
    case uploaded
    case starting
    case running
    case succeeded
    case failed
    case downloading
    case downloaded
    case imported

    var isBusy: Bool {
        switch self {
        case .uploading, .starting, .downloading:
            return true
        default:
            return false
        }
    }

    var isTerminal: Bool {
        switch self {
        case .succeeded, .failed, .downloaded, .imported:
            return true
        default:
            return false
        }
    }
}

struct RunPodRemoteJobState: Equatable, Sendable {
    var archivePath: String
    var phase: RunPodRemoteJobPhase
    var message: String
    var logTail: String
    var result: RemoteTrainingJobResult?
    var localArtifactsPath: String?
    var importedAdapterName: String?
}

struct RunPodRemoteJobPullback: Sendable {
    var directoryURL: URL
    var artifactsURL: URL
    var finalAdapterURL: URL
    var result: RemoteTrainingJobResult
}

enum RunPodRemoteJobError: LocalizedError {
    case invalidConfiguration(String)
    case commandFailed(String)
    case invalidResult(String)
    case timedOut(String)

    var errorDescription: String? {
        switch self {
        case .invalidConfiguration(let detail):
            return "Remote worker configuration is incomplete: \(detail)"
        case .commandFailed(let detail):
            return "Remote worker command failed: \(detail)"
        case .invalidResult(let detail):
            return "Remote worker result is invalid: \(detail)"
        case .timedOut(let command):
            return "\(command) timed out."
        }
    }

    static func userFacingMessage(for error: Error) -> String {
        if let localized = error as? LocalizedError,
           let description = localized.errorDescription {
            return description
        }
        return error.localizedDescription
    }
}

struct RunPodRemoteCommand: Equatable, Sendable {
    var executableURL: URL
    var arguments: [String]
    var displayName: String
}

struct RunPodRemoteCommandResult: Equatable, Sendable {
    var exitStatus: Int32
    var stdout: String
    var stderr: String
}

protocol RunPodRemoteCommandExecuting: Sendable {
    func run(
        executableURL: URL,
        arguments: [String],
        timeoutSeconds: TimeInterval
    ) async throws -> RunPodRemoteCommandResult
}

enum RunPodSSHKeyStoreError: LocalizedError {
    case invalidPrivateKey
    case keyTooLarge

    var errorDescription: String? {
        switch self {
        case .invalidPrivateKey:
            return "Choose an SSH private key, not a public key or another file."
        case .keyTooLarge:
            return "The selected SSH private key is unexpectedly large."
        }
    }
}

struct RunPodSSHKeyStore {
    private static let maximumPrivateKeySize = 128 * 1_024
    private static let supportedPrivateKeyHeaders = [
        "-----BEGIN OPENSSH PRIVATE KEY-----",
        "-----BEGIN PRIVATE KEY-----",
        "-----BEGIN RSA PRIVATE KEY-----",
        "-----BEGIN EC PRIVATE KEY-----",
        "-----BEGIN DSA PRIVATE KEY-----",
    ]

    let managedDirectoryURL: URL

    init(
        managedDirectoryURL: URL = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".ssh", isDirectory: true)
    ) {
        self.managedDirectoryURL = managedDirectoryURL
    }

    var managedPrivateKeyURL: URL {
        managedDirectoryURL.appendingPathComponent(
            "id_ed25519",
            isDirectory: false
        )
    }

    var hasManagedPrivateKey: Bool {
        guard let attributes = try? FileManager.default.attributesOfItem(
            atPath: managedPrivateKeyURL.path(percentEncoded: false)
        ) else {
            return false
        }
        return attributes[.type] as? FileAttributeType == .typeRegular
    }

    @discardableResult
    func importPrivateKey(from sourceURL: URL) throws -> URL {
        let didStartAccessing = sourceURL.startAccessingSecurityScopedResource()
        defer {
            if didStartAccessing {
                sourceURL.stopAccessingSecurityScopedResource()
            }
        }

        let resourceValues = try sourceURL.resourceValues(
            forKeys: [.fileSizeKey, .isRegularFileKey]
        )
        guard resourceValues.isRegularFile == true else {
            throw RunPodSSHKeyStoreError.invalidPrivateKey
        }
        guard (resourceValues.fileSize ?? 0) <= Self.maximumPrivateKeySize else {
            throw RunPodSSHKeyStoreError.keyTooLarge
        }

        let privateKey = try Data(contentsOf: sourceURL)
        guard !privateKey.isEmpty,
              let privateKeyText = String(data: privateKey, encoding: .utf8),
              Self.supportedPrivateKeyHeaders.contains(where: {
                  privateKeyText.contains($0)
              }) else {
            throw RunPodSSHKeyStoreError.invalidPrivateKey
        }

        try FileManager.default.createDirectory(
            at: managedDirectoryURL,
            withIntermediateDirectories: true,
            attributes: [.posixPermissions: 0o700]
        )
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o700],
            ofItemAtPath: managedDirectoryURL.path(percentEncoded: false)
        )
        try privateKey.write(to: managedPrivateKeyURL, options: .atomic)
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o600],
            ofItemAtPath: managedPrivateKeyURL.path(percentEncoded: false)
        )
        return managedPrivateKeyURL
    }

    func removeManagedPrivateKey() throws {
        guard hasManagedPrivateKey else { return }
        try FileManager.default.removeItem(at: managedPrivateKeyURL)
    }

    static func configuredPrivateKeyURL(for path: String) -> URL? {
        let trimmedPath = path.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedPath.isEmpty else { return nil }
        return URL(
            fileURLWithPath: NSString(string: trimmedPath).expandingTildeInPath,
            isDirectory: false
        )
    }

    static var suggestedSourceDirectoryURL: URL {
        AppStorageLocations.realUserHomeDirectory()
            .appendingPathComponent(".ssh", isDirectory: true)
    }
}

struct RunPodRemoteJobPlan: Sendable {
    let podID: String
    let archiveURL: URL
    let workerDirectoryURL: URL
    let host: String
    let port: Int
    let sshKeyPath: String?
    let remoteJobKey: String

    init(
        pod: RunPodPodSummary,
        archiveURL: URL,
        sshKeyPath: String,
        workerDirectoryURL: URL = RunPodRemoteWorkerLocator.defaultDirectoryURL()
    ) throws {
        var archiveIsDirectory: ObjCBool = false
        guard FileManager.default.fileExists(
            atPath: archiveURL.path(percentEncoded: false),
            isDirectory: &archiveIsDirectory
        ), !archiveIsDirectory.boolValue else {
            throw RunPodRemoteJobError.invalidConfiguration(
                "the selected remote job zip no longer exists"
            )
        }
        guard FileManager.default.fileExists(
            atPath: workerDirectoryURL
                .appendingPathComponent("install.sh")
                .path(percentEncoded: false)
        ) else {
            throw RunPodRemoteJobError.invalidConfiguration(
                "the portable PiMP remote worker could not be found"
            )
        }
        guard let publicIP = pod.publicIp?
                .trimmingCharacters(in: .whitespacesAndNewlines),
              !publicIP.isEmpty,
              publicIP.rangeOfCharacter(
                from: CharacterSet(
                    charactersIn: "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-"
                ).inverted
              ) == nil,
              let sshPort = pod.sshPort,
              (1...65_535).contains(sshPort) else {
            throw RunPodRemoteJobError.invalidConfiguration(
                "the pod does not expose a valid public SSH endpoint"
            )
        }

        let expandedKey = RunPodSSHKeyStore
            .configuredPrivateKeyURL(for: sshKeyPath)?
            .path(percentEncoded: false)
        if let expandedKey,
           !FileManager.default.fileExists(atPath: expandedKey) {
            throw RunPodRemoteJobError.invalidConfiguration(
                "SSH key not found at \(expandedKey). Choose the private key in the RunPod section."
            )
        }

        self.podID = pod.id
        self.archiveURL = archiveURL
        self.workerDirectoryURL = workerDirectoryURL
        self.host = publicIP
        self.port = sshPort
        self.sshKeyPath = expandedKey
        self.remoteJobKey = Self.safeRemoteJobKey(
            archiveURL.deletingPathExtension().lastPathComponent
        )
    }

    var prepareUploadCommand: RunPodRemoteCommand {
        sshCommand(
            """
            set -euo pipefail
            mkdir -p /workspace/jobs
            rm -rf /workspace/pimp-remote-worker-src
            """,
            displayName: "Prepare remote upload"
        )
    }

    var uploadArchiveCommand: RunPodRemoteCommand {
        scpCommand(
            source: archiveURL.path(percentEncoded: false),
            destination: "root@\(host):/workspace/jobs/",
            recursive: false,
            displayName: "Upload remote job zip"
        )
    }

    var uploadWorkerCommand: RunPodRemoteCommand {
        scpCommand(
            source: workerDirectoryURL.path(percentEncoded: false),
            destination: "root@\(host):/workspace/pimp-remote-worker-src",
            recursive: true,
            displayName: "Upload portable remote worker"
        )
    }

    var startCommand: RunPodRemoteCommand {
        let root = remoteRoot
        let archive = "/workspace/jobs/\(archiveURL.lastPathComponent)"
        return sshCommand(
            """
            set -euo pipefail
            ROOT=\(shellQuoted(root))
            ARCHIVE=\(shellQuoted(archive))
            rm -rf "$ROOT"
            mkdir -p "$ROOT"
            date +%s > "$ROOT/setup.started"
            if [ -f /workspace/pimp-remote-worker/env.sh ]; then
              echo "refresh" > "$ROOT/setup.kind"
            else
              echo "initial" > "$ROOT/setup.kind"
            fi
            touch "$ROOT/setup.in-progress"
            rm -f "$ROOT/launcher.failed"
            cat > "$ROOT/launch.sh" <<'PIMP_LAUNCHER'
            #!/usr/bin/env bash
            set -euo pipefail
            ROOT="${PIMP_LAUNCH_ROOT:?}"
            ARCHIVE="${PIMP_LAUNCH_ARCHIVE:?}"
            finish() {
              STATUS=$?
              rm -f "$ROOT/setup.in-progress"
              if [ "$STATUS" -ne 0 ]; then
                printf '%s\n' "$STATUS" > "$ROOT/launcher.failed"
              fi
            }
            trap finish EXIT
            if [ -f /workspace/pimp-remote-worker-src/install.sh ]; then
              bash /workspace/pimp-remote-worker-src/install.sh > "$ROOT/install.log" 2>&1
            fi
            if [ -f /workspace/pimp-remote-worker/env.sh ]; then
              source /workspace/pimp-remote-worker/env.sh
            fi
            if ! command -v pimp-remote-worker >/dev/null 2>&1; then
              echo "The portable PiMP remote worker was not installed." >&2
              exit 127
            fi
            export PIMP_REMOTE_JOB_DIR="$ROOT"
            nohup pimp-remote-worker train "$ARCHIVE" \
              > "$ROOT/console.log" 2>&1 < /dev/null &
            WORKER_PID=$!
            printf '%s\n' "$WORKER_PID" > "$ROOT/worker.pid"
            echo "Started PiMP remote worker (PID $WORKER_PID)."
            PIMP_LAUNCHER
            chmod 0755 "$ROOT/launch.sh"
            nohup env \
              PIMP_LAUNCH_ROOT="$ROOT" \
              PIMP_LAUNCH_ARCHIVE="$ARCHIVE" \
              bash "$ROOT/launch.sh" \
              > "$ROOT/launcher.log" 2>&1 < /dev/null &
            LAUNCHER_PID=$!
            printf '%s\n' "$LAUNCHER_PID" > "$ROOT/launcher.pid"
            echo "Remote setup started (PID $LAUNCHER_PID). First setup usually takes 20\u{2013}45 minutes."
            """,
            displayName: "Start remote worker"
        )
    }

    var statusCommand: RunPodRemoteCommand {
        let root = remoteRoot
        return sshCommand(
            """
            set -euo pipefail
            ROOT=\(shellQuoted(root))
            RESULT="$(find "$ROOT" -maxdepth 8 -path '*/remote_artifacts/result.json' -type f -print -quit)"
            LOGS="$(find "$ROOT" -maxdepth 8 -path '*/remote_artifacts/logs.jsonl' -type f -print -quit)"
            CONSOLE="$ROOT/console.log"
            PID_FILE="$ROOT/worker.pid"
            INSTALL_LOG="$ROOT/install.log"
            SETUP_STARTED="$ROOT/setup.started"
            SETUP_KIND="$ROOT/setup.kind"
            SETUP_ACTIVE="$ROOT/setup.in-progress"
            LAUNCHER_PID_FILE="$ROOT/launcher.pid"
            LAUNCHER_FAILED="$ROOT/launcher.failed"
            LAUNCHER_LOG="$ROOT/launcher.log"
            if [ -n "$RESULT" ]; then
              echo "PIMP_STATUS:RESULT"
              cat "$RESULT"
            elif [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
              echo "PIMP_STATUS:RUNNING"
              if [ -n "$LOGS" ]; then tail -n 30 "$LOGS"; elif [ -f "$CONSOLE" ]; then tail -n 30 "$CONSOLE"; fi
            elif [ -f "$PID_FILE" ]; then
              echo "PIMP_STATUS:FAILED"
              if [ -n "$LOGS" ]; then tail -n 30 "$LOGS"; elif [ -f "$CONSOLE" ]; then tail -n 30 "$CONSOLE"; fi
            elif [ -f "$LAUNCHER_FAILED" ]; then
              echo "PIMP_STATUS:FAILED"
              echo "Remote setup failed with status $(cat "$LAUNCHER_FAILED")."
              if [ -f "$INSTALL_LOG" ]; then tail -n 30 "$INSTALL_LOG"; elif [ -f "$LAUNCHER_LOG" ]; then tail -n 30 "$LAUNCHER_LOG"; fi
            elif [ -f "$LAUNCHER_PID_FILE" ] \
              && ! kill -0 "$(cat "$LAUNCHER_PID_FILE")" 2>/dev/null \
              && [ -f "$SETUP_ACTIVE" ]; then
              echo "PIMP_STATUS:FAILED"
              echo "The remote setup process stopped before starting the worker."
              if [ -f "$INSTALL_LOG" ]; then tail -n 30 "$INSTALL_LOG"; elif [ -f "$LAUNCHER_LOG" ]; then tail -n 30 "$LAUNCHER_LOG"; fi
            else
              echo "PIMP_STATUS:PENDING"
              if [ -f "$SETUP_ACTIVE" ]; then
                NOW="$(date +%s)"
                STARTED="$(cat "$SETUP_STARTED" 2>/dev/null || echo "$NOW")"
                ELAPSED_SECONDS=$((NOW - STARTED))
                if [ "$ELAPSED_SECONDS" -lt 0 ]; then ELAPSED_SECONDS=0; fi
                ELAPSED_MINUTES=$((ELAPSED_SECONDS / 60))
                STAGE="Preparing the remote environment"
                if [ -f "$INSTALL_LOG" ]; then
                  if grep -q "\\[PiMP\\] installing training dependencies" "$INSTALL_LOG"; then
                    STAGE="Installing PyTorch, CUDA, and training dependencies"
                  elif grep -q "\\[PiMP\\] creating Python environment" "$INSTALL_LOG"; then
                    STAGE="Creating the Python environment"
                  elif grep -q "\\[PiMP\\] cloning ACE-Step" "$INSTALL_LOG"; then
                    STAGE="Downloading the ACE-Step training code"
                  elif grep -q "\\[PiMP\\] installing system packages" "$INSTALL_LOG"; then
                    STAGE="Installing system packages"
                  fi
                fi
                if [ "$(cat "$SETUP_KIND" 2>/dev/null || echo initial)" = "refresh" ]; then
                  echo "$STAGE. Elapsed: $ELAPSED_MINUTES min; estimated setup time: 2\u{2013}8 min."
                else
                  echo "$STAGE. Elapsed: $ELAPSED_MINUTES min; estimated first setup time: 20\u{2013}45 min. Network-backed storage can take longer."
                fi
              elif [ -f "$INSTALL_LOG" ]; then
                echo "Remote setup stopped before the worker started. Check the Start result and install log."
              else
                echo "Remote launch is initializing. Estimated first setup time: 20\u{2013}45 min."
              fi
            fi
            """,
            displayName: "Refresh remote worker status"
        )
    }

    var preparePullbackCommand: RunPodRemoteCommand {
        let root = remoteRoot
        return sshCommand(
            """
            set -euo pipefail
            ROOT=\(shellQuoted(root))
            RESULT="$(find "$ROOT" -maxdepth 8 -path '*/remote_artifacts/result.json' -type f -print -quit)"
            test -n "$RESULT"
            PACKAGE_ROOT="$(dirname "$(dirname "$RESULT")")"
            case "$PACKAGE_ROOT" in "$ROOT"/*) ;; *) echo "Unsafe package root." >&2; exit 4 ;; esac
            rm -rf "$ROOT/pullback"
            mkdir -p "$ROOT/pullback"
            cp -R "$PACKAGE_ROOT/remote_artifacts" "$ROOT/pullback/remote_artifacts"
            """,
            displayName: "Prepare remote artifact pullback"
        )
    }

    func pullArtifactsCommand(destinationURL: URL) -> RunPodRemoteCommand {
        scpCommand(
            source: "root@\(host):\(remoteRoot)/pullback/remote_artifacts",
            destination: destinationURL.path(percentEncoded: false),
            recursive: true,
            displayName: "Download remote artifacts"
        )
    }

    func makeLocalPullbackDirectory() throws -> URL {
        let parent = archiveURL.deletingLastPathComponent()
            .appendingPathComponent("Remote Training Results", isDirectory: true)
        try FileManager.default.createDirectory(
            at: parent,
            withIntermediateDirectories: true
        )
        let podSuffix = String(podID.prefix(8))
        var destination = parent.appendingPathComponent(
            "\(remoteJobKey)-\(podSuffix)",
            isDirectory: true
        )
        if FileManager.default.fileExists(
            atPath: destination.path(percentEncoded: false)
        ) {
            destination = parent.appendingPathComponent(
                "\(remoteJobKey)-\(podSuffix)-\(UUID().uuidString.prefix(8))",
                isDirectory: true
            )
        }
        try FileManager.default.createDirectory(
            at: destination,
            withIntermediateDirectories: true
        )
        return destination
    }

    private var remoteRoot: String {
        "/workspace/pimp-remote-jobs/\(remoteJobKey)"
    }

    private func sshCommand(
        _ remoteCommand: String,
        displayName: String
    ) -> RunPodRemoteCommand {
        RunPodRemoteCommand(
            executableURL: URL(fileURLWithPath: "/usr/bin/ssh"),
            arguments: sshOptions
                + ["-p", String(port), "root@\(host)", remoteCommand],
            displayName: displayName
        )
    }

    private func scpCommand(
        source: String,
        destination: String,
        recursive: Bool,
        displayName: String
    ) -> RunPodRemoteCommand {
        var arguments = sshOptions + ["-P", String(port)]
        if recursive {
            arguments.append("-r")
        }
        arguments.append(contentsOf: [source, destination])
        return RunPodRemoteCommand(
            executableURL: URL(fileURLWithPath: "/usr/bin/scp"),
            arguments: arguments,
            displayName: displayName
        )
    }

    private var sshOptions: [String] {
        var arguments = [
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ConnectTimeout=20",
        ]
        if let sshKeyPath {
            arguments.append(contentsOf: ["-i", sshKeyPath])
        }
        return arguments
    }

    private static func safeRemoteJobKey(_ value: String) -> String {
        let normalized = value
            .lowercased()
            .replacingOccurrences(
                of: "[^a-z0-9-]+",
                with: "-",
                options: .regularExpression
            )
            .trimmingCharacters(in: CharacterSet(charactersIn: "-"))
        let base = normalized.isEmpty ? "remote-job" : normalized
        return String(base.prefix(80))
    }
}

enum RunPodRemoteStatus {
    case pending
    case running
    case failed
    case result
}

struct RunPodRemoteStatusSnapshot {
    var status: RunPodRemoteStatus
    var logTail: String
    var result: RemoteTrainingJobResult?

    static func parse(_ output: String) throws -> RunPodRemoteStatusSnapshot {
        let markers: [(String, RunPodRemoteStatus)] = [
            ("PIMP_STATUS:RESULT", .result),
            ("PIMP_STATUS:RUNNING", .running),
            ("PIMP_STATUS:FAILED", .failed),
            ("PIMP_STATUS:PENDING", .pending),
        ]
        guard let match = markers.first(where: { output.contains($0.0) }),
              let markerRange = output.range(of: match.0) else {
            throw RunPodRemoteJobError.invalidResult(
                "status response did not contain a PiMP status marker"
            )
        }
        let payload = output[markerRange.upperBound...]
            .trimmingCharacters(in: .whitespacesAndNewlines)
        if case .result = match.1 {
            guard let data = payload.data(using: .utf8) else {
                throw RunPodRemoteJobError.invalidResult(
                    "result.json was not valid UTF-8"
                )
            }
            let result = try RemoteTrainingJobCodec.decoder.decode(
                RemoteTrainingJobResult.self,
                from: data
            )
            guard result.pimpAudioRemoteResultVersion
                    == RemoteTrainingJobResult.schemaVersion else {
                throw RunPodRemoteJobError.invalidResult(
                    "unsupported result schema \(result.pimpAudioRemoteResultVersion)"
                )
            }
            return RunPodRemoteStatusSnapshot(
                status: .result,
                logTail: "",
                result: result
            )
        }
        return RunPodRemoteStatusSnapshot(
            status: match.1,
            logTail: payload,
            result: nil
        )
    }
}

final class RunPodRemoteProcessExecutor:
    RunPodRemoteCommandExecuting,
    @unchecked Sendable
{
    func run(
        executableURL: URL,
        arguments: [String],
        timeoutSeconds: TimeInterval
    ) async throws -> RunPodRemoteCommandResult {
        let process = RunPodCapturedProcess(
            executableURL: executableURL,
            arguments: arguments
        )
        return try await process.run(timeoutSeconds: timeoutSeconds)
    }
}

private final class RunPodCapturedProcess: @unchecked Sendable {
    private let process = Process()
    private let stdout = Pipe()
    private let stderr = Pipe()
    private let lock = NSLock()
    private var stdoutData = Data()
    private var stderrData = Data()
    private var continuation: CheckedContinuation<
        RunPodRemoteCommandResult,
        Error
    >?
    private var didResume = false
    private var didCancel = false

    init(executableURL: URL, arguments: [String]) {
        process.executableURL = executableURL
        process.arguments = arguments
        process.standardOutput = stdout
        process.standardError = stderr
    }

    func run(timeoutSeconds: TimeInterval) async throws -> RunPodRemoteCommandResult {
        try await withThrowingTaskGroup(
            of: RunPodRemoteCommandResult.self
        ) { group in
            group.addTask {
                try await self.runProcess()
            }
            group.addTask {
                try await Task.sleep(
                    nanoseconds: UInt64(timeoutSeconds * 1_000_000_000)
                )
                self.cancel()
                throw RunPodRemoteJobError.timedOut(
                    self.process.executableURL?.lastPathComponent ?? "remote command"
                )
            }
            guard let result = try await group.next() else {
                throw CancellationError()
            }
            group.cancelAll()
            return result
        }
    }

    private func runProcess() async throws -> RunPodRemoteCommandResult {
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation {
                (continuation: CheckedContinuation<
                    RunPodRemoteCommandResult,
                    Error
                >) in
                start(continuation)
            }
        } onCancel: {
            cancel()
        }
    }

    private func start(
        _ continuation: CheckedContinuation<
            RunPodRemoteCommandResult,
            Error
        >
    ) {
        lock.lock()
        if didCancel {
            lock.unlock()
            continuation.resume(throwing: CancellationError())
            return
        }
        self.continuation = continuation
        lock.unlock()

        stdout.fileHandleForReading.readabilityHandler = { [weak self] handle in
            self?.append(handle.availableData, toStdout: true)
        }
        stderr.fileHandleForReading.readabilityHandler = { [weak self] handle in
            self?.append(handle.availableData, toStdout: false)
        }
        process.terminationHandler = { [weak self] process in
            self?.finish(process)
        }
        do {
            try process.run()
        } catch {
            finish(error)
        }
    }

    private func append(_ data: Data, toStdout: Bool) {
        guard !data.isEmpty else { return }
        lock.lock()
        if toStdout {
            stdoutData.append(data)
        } else {
            stderrData.append(data)
        }
        lock.unlock()
    }

    private func finish(_ process: Process) {
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
        let output = stdoutData
        let errors = stderrData
        lock.unlock()
        continuation.resume(returning: RunPodRemoteCommandResult(
            exitStatus: process.terminationStatus,
            stdout: String(data: output, encoding: .utf8) ?? "",
            stderr: String(data: errors, encoding: .utf8) ?? ""
        ))
    }

    private func finish(_ error: Error) {
        stdout.fileHandleForReading.readabilityHandler = nil
        stderr.fileHandleForReading.readabilityHandler = nil
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
        lock.lock()
        didCancel = true
        let continuation = didResume ? nil : self.continuation
        if continuation != nil {
            didResume = true
            self.continuation = nil
        }
        lock.unlock()
        if process.isRunning {
            process.terminate()
        }
        continuation?.resume(throwing: CancellationError())
    }
}

struct RunPodLaunchConfiguration: Equatable, Sendable {
    var name: String
    var templateID: String
    var imageName: String
    var gpuTypeID: String
    var interruptible: Bool
    var containerDiskGB: Int
    var volumeGB: Int
    var ports: [String] = ["22/tcp", "8888/http"]
    var supportPublicIP: Bool = true

    func requestBody() throws -> RunPodCreatePodRequest {
        let trimmedName = name.trimmingCharacters(in: .whitespacesAndNewlines)
        let trimmedTemplateID = templateID.trimmingCharacters(in: .whitespacesAndNewlines)
        let trimmedImageName = imageName.trimmingCharacters(in: .whitespacesAndNewlines)
        let trimmedGPUType = gpuTypeID.trimmingCharacters(in: .whitespacesAndNewlines)

        guard !trimmedTemplateID.isEmpty || !trimmedImageName.isEmpty else {
            throw RunPodError.invalidConfiguration("Enter a RunPod template ID or container image.")
        }
        guard !trimmedGPUType.isEmpty else {
            throw RunPodError.invalidConfiguration("Enter at least one RunPod GPU type.")
        }
        guard containerDiskGB >= 20 else {
            throw RunPodError.invalidConfiguration("Container disk must be at least 20 GB.")
        }
        guard volumeGB >= 20 else {
            throw RunPodError.invalidConfiguration("Volume must be at least 20 GB.")
        }

        let requestTemplateID = trimmedImageName.isEmpty ? trimmedTemplateID : ""
        let requestImageName = trimmedImageName

        return RunPodCreatePodRequest(
            cloudType: "SECURE",
            computeType: "GPU",
            containerDiskInGb: containerDiskGB,
            gpuCount: 1,
            gpuTypeIds: [trimmedGPUType],
            gpuTypePriority: "availability",
            imageName: requestImageName.isEmpty ? nil : requestImageName,
            interruptible: interruptible,
            locked: false,
            name: trimmedName.isEmpty ? "PiMP Audio Studio Remote Job" : trimmedName,
            ports: ports,
            supportPublicIp: supportPublicIP,
            templateId: requestTemplateID.isEmpty ? nil : requestTemplateID,
            volumeInGb: volumeGB,
            volumeMountPath: "/workspace"
        )
    }

    var diagnosticsSummary: String {
        [
            "name=\(name)",
            "templateID=\(templateID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? "none" : templateID)",
            "image=\(imageName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? "none" : imageName)",
            "gpu=\(gpuTypeID)",
            "interruptible=\(interruptible)",
            "containerDiskGB=\(containerDiskGB)",
            "volumeGB=\(volumeGB)"
        ].joined(separator: " ")
    }
}

struct RunPodPodSummary: Identifiable, Decodable, Equatable, Sendable {
    var id: String
    var name: String?
    var desiredStatus: String?
    var image: String?
    var interruptible: Bool?
    var locked: Bool?
    var costPerHr: RunPodFlexibleDouble?
    var adjustedCostPerHr: RunPodFlexibleDouble?
    var gpu: RunPodGPU?
    var machine: RunPodMachine?
    var gpuCount: Int?
    var memoryInGb: Double?
    var publicIp: String?
    var portMappings: [String: Int]
    var ports: [String]
    var templateId: String?
    var lastStartedAt: String?
    var lastStatusChange: String?
    var containerDiskInGb: Int?
    var volumeInGb: Int?
    var volumeMountPath: String?

    enum CodingKeys: String, CodingKey {
        case id
        case name
        case desiredStatus
        case image
        case interruptible
        case locked
        case costPerHr
        case adjustedCostPerHr
        case gpu
        case machine
        case gpuCount
        case memoryInGb
        case publicIp
        case portMappings
        case ports
        case templateId
        case lastStartedAt
        case lastStatusChange
        case containerDiskInGb
        case volumeInGb
        case volumeMountPath
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(String.self, forKey: .id)
        name = try container.decodeIfPresent(String.self, forKey: .name)
        desiredStatus = try container.decodeIfPresent(String.self, forKey: .desiredStatus)
        image = try container.decodeIfPresent(String.self, forKey: .image)
        interruptible = try container.decodeIfPresent(Bool.self, forKey: .interruptible)
        locked = try container.decodeIfPresent(Bool.self, forKey: .locked)
        costPerHr = try container.decodeIfPresent(RunPodFlexibleDouble.self, forKey: .costPerHr)
        adjustedCostPerHr = try container.decodeIfPresent(RunPodFlexibleDouble.self, forKey: .adjustedCostPerHr)
        gpu = try container.decodeIfPresent(RunPodGPU.self, forKey: .gpu)
        machine = try container.decodeIfPresent(RunPodMachine.self, forKey: .machine)
        gpuCount = try container.decodeIfPresent(Int.self, forKey: .gpuCount)
        memoryInGb = try container.decodeIfPresent(Double.self, forKey: .memoryInGb)
        publicIp = try container.decodeIfPresent(String.self, forKey: .publicIp)
        portMappings = (try? container.decode([String: Int].self, forKey: .portMappings)) ?? [:]
        ports = (try? container.decode([String].self, forKey: .ports)) ?? []
        templateId = try container.decodeIfPresent(String.self, forKey: .templateId)
        lastStartedAt = try container.decodeIfPresent(String.self, forKey: .lastStartedAt)
        lastStatusChange = try container.decodeIfPresent(String.self, forKey: .lastStatusChange)
        containerDiskInGb = try container.decodeIfPresent(Int.self, forKey: .containerDiskInGb)
        volumeInGb = try container.decodeIfPresent(Int.self, forKey: .volumeInGb)
        volumeMountPath = try container.decodeIfPresent(String.self, forKey: .volumeMountPath)
    }

    var displayName: String {
        let trimmed = (name ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? id : trimmed
    }

    var statusLabel: String {
        desiredStatus ?? "UNKNOWN"
    }

    var gpuLabel: String {
        gpu?.displayName ?? machine?.gpuDisplayName ?? machine?.gpuType?.displayName ?? "GPU"
    }

    var hourlyCostLabel: String? {
        let value = adjustedCostPerHr?.value ?? costPerHr?.value
        guard let value else { return nil }
        return String(format: "$%.2f/hr", value)
    }

    var sshPort: Int? {
        portMappings["22"]
    }

    func sshCommand(sshKeyPath: String) -> String? {
        guard let publicIp, !publicIp.isEmpty, let sshPort else { return nil }
        let key = sshKeyPath.trimmingCharacters(in: .whitespacesAndNewlines)
        let keyArgument = key.isEmpty ? "" : " -i \(shellPathQuoted(key))"
        return "ssh -p \(sshPort)\(keyArgument) root@\(publicIp)"
    }

    func uploadCommand(
        archiveURL: URL?,
        sshKeyPath: String,
        workerDirectoryURL: URL? = RunPodRemoteWorkerLocator.defaultDirectoryURL()
    ) -> String? {
        guard let archiveURL,
              let publicIp,
              !publicIp.isEmpty,
              let sshPort else { return nil }
        let key = sshKeyPath.trimmingCharacters(in: .whitespacesAndNewlines)
        let keyArgument = key.isEmpty ? "" : " -i \(shellPathQuoted(key))"
        let workerDirectory = shellQuoted(
            (workerDirectoryURL ?? RunPodRemoteWorkerLocator.defaultDirectoryURL())
                .path(percentEncoded: false)
        )
        let makeJobsDirectory = "ssh -p \(sshPort)\(keyArgument) root@\(publicIp) 'mkdir -p /workspace/jobs && rm -rf /workspace/pimp-remote-worker-src'"
        let uploadArchive = "scp -P \(sshPort)\(keyArgument) \(shellPathQuoted(archiveURL.path(percentEncoded: false))) root@\(publicIp):/workspace/jobs/"
        let uploadWorker = "scp -P \(sshPort)\(keyArgument) -r \"$WORKER_DIR\" root@\(publicIp):/workspace/pimp-remote-worker-src"
        return """
        (
        set -euo pipefail
        WORKER_DIR=\(workerDirectory)
        if [ ! -f "$WORKER_DIR/install.sh" ]; then
          echo "Could not find PiMP remote worker at $WORKER_DIR"
          echo "Set PIMP_REMOTE_WORKER_DIR=/path/to/remote-worker or rebuild with the worker resource bundled."
          exit 2
        fi
        \(makeJobsDirectory) && \\
        \(uploadArchive) && \\
        \(uploadWorker)
        )
        """
    }

    func remotePackageCommand(archiveURL: URL?) -> String? {
        guard let archiveURL else { return nil }
        let zipName = shellQuoted(archiveURL.lastPathComponent)
        return """
        bash <<'PIMP_REMOTE_RUN'
        set -euo pipefail
        cd /workspace/jobs
        ZIP=\(zipName)
        if [ -f /workspace/pimp-remote-worker-src/install.sh ]; then
          bash /workspace/pimp-remote-worker-src/install.sh
        fi
        if [ -f /workspace/pimp-remote-worker/env.sh ]; then
          source /workspace/pimp-remote-worker/env.sh
        fi
        if ! command -v pimp-remote-worker >/dev/null 2>&1; then
          echo "The portable PiMP remote worker was not installed."
          exit 127
        fi
        export PIMP_REMOTE_JOB_DIR=/workspace/pimp-remote-jobs/current
        pimp-remote-worker train "$ZIP"
        PIMP_REMOTE_RUN
        """
    }

    var diagnosticsSummary: String {
        [
            "id=\(id)",
            "name=\(displayName)",
            "status=\(statusLabel)",
            "gpu=\(gpuLabel)",
            "cost=\(hourlyCostLabel ?? "unknown")",
            "ssh=\(sshPort == nil ? "unavailable" : "available")"
        ].joined(separator: " ")
    }

}

struct RunPodGPU: Decodable, Equatable, Sendable {
    var id: String?
    var count: Int?
    var displayName: String?
    var securePrice: RunPodFlexibleDouble?
    var secureSpotPrice: RunPodFlexibleDouble?
}

struct RunPodGPUType: Identifiable, Decodable, Equatable, Sendable {
    var id: String
    var displayName: String?
    var manufacturer: String?
    var memoryInGb: Int?
    var secureCloud: Bool?
    var secureLowestPrice: RunPodLowestPrice?
    var securePrice: RunPodFlexibleDouble?
    var secureSpotPrice: RunPodFlexibleDouble?
    var maxGpuCount: Int?
    var maxGpuCountSecureCloud: Int?

    var selectionValue: String {
        let trimmedID = id.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmedID.isEmpty {
            return trimmedID
        }
        return displayNameValue
    }

    var displayNameValue: String {
        let trimmed = (displayName ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? id : trimmed
    }

    var menuTitle: String {
        var details: [String] = []
        if let memoryInGb {
            details.append("\(memoryInGb) GB")
        }
        if let liveStockLabel {
            details.append(liveStockLabel)
        }
        if let lowestHourlyPriceLabel {
            details.append("from \(lowestHourlyPriceLabel)")
        }
        if details.isEmpty {
            return displayNameValue
        }
        return "\(displayNameValue) (\(details.joined(separator: ", ")))"
    }

    func detailLabel(interruptible: Bool) -> String {
        var details: [String] = []
        if let memoryInGb {
            details.append("\(memoryInGb) GB VRAM")
        }
        details.append(availabilityLabel)
        details.append(stockLabel())
        if let launchPriceLabel = launchPriceLabel(interruptible: interruptible) {
            details.append("\(interruptible ? "Spot" : "On-demand") \(launchPriceLabel)")
        }
        if let maxGpuCount = maxGpuCountSecureCloud ?? maxGpuCount, maxGpuCount > 0 {
            details.append("up to \(maxGpuCount) GPU\(maxGpuCount == 1 ? "" : "s")")
        }
        return details.joined(separator: " · ")
    }

    var availabilityLabel: String {
        if secureCloud == false {
            return "Secure Cloud unavailable"
        }
        return "Secure Cloud"
    }

    var liveStockLabel: String? {
        secureLowestPrice?.stockStatusLabel
    }

    var lowestHourlyPriceLabel: String? {
        let prices = [
            secureLowestPrice?.minimumBidPrice?.value,
            secureLowestPrice?.uninterruptablePrice?.value,
            securePrice?.value,
            secureSpotPrice?.value,
        ].compactMap { $0 }.filter { $0 > 0 }

        guard let lowest = prices.min() else { return nil }
        return String(format: "$%.2f/hr", lowest)
    }

    func matches(_ value: String) -> Bool {
        let normalized = value.trimmingCharacters(in: .whitespacesAndNewlines)
        return normalized == id || normalized == displayNameValue
    }

    private func stockLabel() -> String {
        guard let price = secureLowestPrice else {
            return "Live stock unknown"
        }

        var parts = [price.stockStatusLabel ?? "Live stock unknown"]
        if let countLabel = price.availableGpuCountsLabel {
            parts.append(countLabel)
        }
        return parts.joined(separator: " · ")
    }

    private func launchPriceLabel(interruptible: Bool) -> String? {
        if let price = secureLowestPrice?.hourlyPriceLabel(interruptible: interruptible) {
            return price
        }

        let fallback = interruptible ? secureSpotPrice ?? securePrice : securePrice
        guard let value = fallback?.value, value > 0 else { return nil }
        return String(format: "$%.2f/hr", value)
    }
}

struct RunPodTemplate: Identifiable, Decodable, Equatable, Sendable {
    var id: String
    var name: String?
    var imageName: String?
    var category: String?
    var isRunpod: Bool?
    var isPublic: Bool?
    var isServerless: Bool?
    var containerDiskInGb: Int?
    var volumeInGb: Int?
    var volumeMountPath: String?
    var ports: [String]?

    var displayNameValue: String {
        let trimmed = (name ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? id : trimmed
    }

    var imageNameValue: String? {
        let trimmed = (imageName ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }

    var isOfficialPodTemplate: Bool {
        isRunpod == true && isServerless != true
    }

    var summaryLabel: String {
        if let imageNameValue {
            return "\(displayNameValue) · \(imageNameValue)"
        }
        return displayNameValue
    }

    var detailLabel: String {
        var details: [String] = []
        if isRunpod == true {
            details.append("Official RunPod")
        }
        if let containerDiskInGb {
            details.append("container \(containerDiskInGb) GB")
        }
        if let volumeInGb {
            details.append("volume \(volumeInGb) GB")
        }
        if let ports, !ports.isEmpty {
            details.append(ports.joined(separator: ", "))
        }
        return details.joined(separator: " · ")
    }

    func matchesTemplateID(_ value: String) -> Bool {
        id == value.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    static func recommendedTrainingTemplate(
        for gpuTypeValue: String,
        from templates: [RunPodTemplate]
    ) -> RunPodTemplate? {
        guard supportsNVIDIATrainingTemplate(gpuTypeValue) else { return nil }

        return templates
            .filter { $0.isOfficialPodTemplate }
            .compactMap { template -> (template: RunPodTemplate, score: Int)? in
                guard let score = template.trainingTemplateScore(for: gpuTypeValue) else {
                    return nil
                }
                return (template, score)
            }
            .sorted { lhs, rhs in
                if lhs.score != rhs.score {
                    return lhs.score > rhs.score
                }
                return lhs.template.displayNameValue.localizedStandardCompare(
                    rhs.template.displayNameValue
                ) == .orderedAscending
            }
            .first?
            .template
    }

    func incompatibilityReason(for gpuTypeValue: String) -> String? {
        let searchable = runtimeSearchableText
        guard isOfficialPodTemplate else {
            return "This is not an official RunPod pod template."
        }
        guard searchable.contains("pytorch") else {
            return "ACE-Step remote training requires a PyTorch template."
        }
        guard !searchable.contains("cluster") else {
            return "Cluster templates are not intended for this single-GPU worker."
        }

        guard Self.isBlackwellGPU(gpuTypeValue) else {
            return nil
        }
        guard let cudaVersion else {
            return "This template does not identify a CUDA version; Blackwell requires CUDA 12.8 or newer."
        }
        guard cudaVersion >= RunPodRuntimeVersion(major: 12, minor: 8) else {
            return "Blackwell requires CUDA 12.8 or newer; this template uses CUDA \(cudaVersion.label)."
        }
        if let pytorchVersion,
           pytorchVersion < RunPodRuntimeVersion(major: 2, minor: 7) {
            return "Blackwell requires PyTorch 2.7 or newer; this template uses PyTorch \(pytorchVersion.label)."
        }
        return nil
    }

    private static func supportsNVIDIATrainingTemplate(_ gpuTypeValue: String) -> Bool {
        let normalized = gpuTypeValue.lowercased()
        guard !normalized.contains("amd") && !normalized.contains("mi300") else {
            return false
        }
        return normalized.contains("nvidia")
            || normalized.contains("rtx")
            || normalized.contains("geforce")
            || normalized.contains("a100")
            || normalized.contains("h100")
            || normalized.contains("h200")
            || normalized.contains("l40")
            || normalized.contains("l4")
            || normalized.contains("tesla")
    }

    private static func isBlackwellGPU(_ gpuTypeValue: String) -> Bool {
        let normalized = gpuTypeValue.lowercased()
        return normalized.contains("blackwell")
            || normalized.contains("geforce rtx 50")
    }

    private func trainingTemplateScore(for gpuTypeValue: String) -> Int? {
        guard incompatibilityReason(for: gpuTypeValue) == nil else { return nil }

        let searchable = runtimeSearchableText
        var score = 1_000
        if searchable.contains("runpod/pytorch") { score += 500 }
        if searchable.contains("runpod pytorch") { score += 500 }
        if let pytorchVersion {
            score += pytorchVersion.major * 1_000
            score += pytorchVersion.minor * 100
            score += min(pytorchVersion.patch, 99)
        }
        if let cudaVersion {
            score += cudaVersion.major * 100
            score += cudaVersion.minor * 10
            score += min(cudaVersion.patch, 9)
        }
        if searchable.contains("devel") { score += 100 }
        if searchable.contains("py3.11") { score += 100 }
        if searchable.contains("ubuntu24.04") || searchable.contains("ubuntu2404") {
            score += 100
        } else if searchable.contains("ubuntu22.04") || searchable.contains("ubuntu2204") {
            score += 75
        }
        return score
    }

    private var runtimeSearchableText: String {
        "\(displayNameValue) \(imageNameValue ?? "")".lowercased()
    }

    private var pytorchVersion: RunPodRuntimeVersion? {
        let name = displayNameValue.lowercased()
        if let version = Self.dottedVersion(after: "pytorch ", in: name) {
            return version
        }
        return Self.compactPyTorchVersion(in: runtimeSearchableText)
    }

    private var cudaVersion: RunPodRuntimeVersion? {
        if let version = Self.dottedVersion(after: "cuda", in: runtimeSearchableText) {
            return version
        }
        return Self.compactCUDAVersion(in: runtimeSearchableText)
    }

    private static func dottedVersion(
        after marker: String,
        in value: String
    ) -> RunPodRuntimeVersion? {
        var searchStart = value.startIndex
        while searchStart < value.endIndex,
              let markerRange = value.range(
                of: marker,
                range: searchStart..<value.endIndex
              ) {
            let suffix = value[markerRange.upperBound...]
            let token = suffix.prefix { $0.isNumber || $0 == "." }
            let components = token
                .split(separator: ".")
                .compactMap { Int($0) }
            if components.count >= 2 {
                return RunPodRuntimeVersion(
                    major: components[0],
                    minor: components[1],
                    patch: components.count > 2 ? components[2] : 0
                )
            }
            searchStart = markerRange.upperBound
        }
        return nil
    }

    private static func compactPyTorchVersion(
        in value: String
    ) -> RunPodRuntimeVersion? {
        var searchStart = value.startIndex
        while searchStart < value.endIndex,
              let markerRange = value.range(
                of: "torch",
                range: searchStart..<value.endIndex
              ) {
            let digits = value[markerRange.upperBound...].prefix { $0.isNumber }
            if digits.count == 3,
               let major = Int(digits.prefix(1)),
               let minor = Int(digits.dropFirst().prefix(1)),
               let patch = Int(digits.suffix(1)) {
                return RunPodRuntimeVersion(
                    major: major,
                    minor: minor,
                    patch: patch
                )
            }
            searchStart = markerRange.upperBound
        }
        return nil
    }

    private static func compactCUDAVersion(
        in value: String
    ) -> RunPodRuntimeVersion? {
        var searchStart = value.startIndex
        while searchStart < value.endIndex,
              let markerRange = value.range(
                of: "cu",
                range: searchStart..<value.endIndex
              ) {
            let digits = value[markerRange.upperBound...].prefix { $0.isNumber }
            if digits.count >= 3,
               let major = Int(digits.prefix(2)),
               let minor = Int(digits.dropFirst(2).prefix(1)) {
                let patchDigits = digits.dropFirst(3)
                return RunPodRuntimeVersion(
                    major: major,
                    minor: minor,
                    patch: Int(patchDigits) ?? 0
                )
            }
            searchStart = markerRange.upperBound
        }
        return nil
    }
}

private struct RunPodRuntimeVersion: Comparable {
    var major: Int
    var minor: Int
    var patch: Int = 0

    var label: String {
        "\(major).\(minor).\(patch)"
    }

    static func < (lhs: RunPodRuntimeVersion, rhs: RunPodRuntimeVersion) -> Bool {
        if lhs.major != rhs.major {
            return lhs.major < rhs.major
        }
        if lhs.minor != rhs.minor {
            return lhs.minor < rhs.minor
        }
        return lhs.patch < rhs.patch
    }
}

struct RunPodLowestPrice: Decodable, Equatable, Sendable {
    var gpuName: String?
    var gpuTypeId: String?
    var minimumBidPrice: RunPodFlexibleDouble?
    var uninterruptablePrice: RunPodFlexibleDouble?
    var stockStatus: String?
    var availableGpuCounts: [Int]?
    var maxGpuCount: Int?
    var maxUnreservedGpuCount: Int?
    var rentedCount: Int?
    var totalCount: Int?

    var stockStatusLabel: String? {
        let trimmed = (stockStatus ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        if trimmed.caseInsensitiveCompare("none") == .orderedSame {
            return "No live stock"
        }
        return "\(trimmed.capitalized) stock"
    }

    var availableGpuCountsLabel: String? {
        let counts = (availableGpuCounts ?? []).filter { $0 > 0 }.sorted()
        guard !counts.isEmpty else { return nil }
        let joined = counts.map(String.init).joined(separator: "/")
        return "counts \(joined)"
    }

    func hourlyPriceLabel(interruptible: Bool) -> String? {
        let price = interruptible
            ? minimumBidPrice?.value ?? uninterruptablePrice?.value
            : uninterruptablePrice?.value ?? minimumBidPrice?.value
        guard let price, price > 0 else { return nil }
        return String(format: "$%.2f/hr", price)
    }
}

struct RunPodMachine: Decodable, Equatable, Sendable {
    var gpuTypeId: String?
    var gpuDisplayName: String?
    var location: String?
    var dataCenterId: String?
    var gpuType: RunPodGPU?
}

struct RunPodFlexibleDouble: Codable, Equatable, Sendable {
    var value: Double

    init(_ value: Double) {
        self.value = value
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let double = try? container.decode(Double.self) {
            value = double
            return
        }
        if let int = try? container.decode(Int.self) {
            value = Double(int)
            return
        }
        if let string = try? container.decode(String.self),
           let double = Double(string) {
            value = double
            return
        }
        throw DecodingError.dataCorruptedError(
            in: container,
            debugDescription: "Expected numeric or string double value"
        )
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(value)
    }
}

struct RunPodCreatePodRequest: Encodable, Equatable, Sendable {
    var cloudType: String
    var computeType: String
    var containerDiskInGb: Int
    var gpuCount: Int
    var gpuTypeIds: [String]
    var gpuTypePriority: String
    var imageName: String?
    var interruptible: Bool
    var locked: Bool
    var name: String
    var ports: [String]
    var supportPublicIp: Bool
    var templateId: String?
    var volumeInGb: Int
    var volumeMountPath: String
}

private struct RunPodGraphQLRequest: Encodable {
    var query: String
}

private struct RunPodGraphQLResponse<Payload: Decodable>: Decodable {
    var data: Payload?
    var errors: [RunPodGraphQLError]?
}

private struct RunPodGraphQLError: Decodable, Equatable {
    var message: String
}

private struct RunPodGPUTypesPayload: Decodable {
    var gpuTypes: [RunPodGPUType]
}

protocol RunPodTransport: Sendable {
    func data(for request: URLRequest) async throws -> (Data, HTTPURLResponse)
}

struct URLSessionRunPodTransport: RunPodTransport {
    var session: URLSession = .shared

    func data(for request: URLRequest) async throws -> (Data, HTTPURLResponse) {
        let (data, response) = try await session.data(for: request)
        guard let httpResponse = response as? HTTPURLResponse else {
            throw RunPodError.invalidResponse
        }
        return (data, httpResponse)
    }
}

struct RunPodAPIClient {
    var baseURL: URL
    var graphQLURL: URL
    var transport: any RunPodTransport

    init(
        baseURL: URL = URL(string: "https://rest.runpod.io/v1")!,
        graphQLURL: URL = URL(string: "https://api.runpod.io/graphql")!,
        transport: any RunPodTransport = URLSessionRunPodTransport()
    ) {
        self.baseURL = baseURL
        self.graphQLURL = graphQLURL
        self.transport = transport
    }

    func listPods(token: String) async throws -> [RunPodPodSummary] {
        let request = try makeRequest(path: "pods", method: "GET", token: token)
        let (data, response) = try await transport.data(for: request)
        try validate(response: response, data: data)
        return try JSONDecoder().decode([RunPodPodSummary].self, from: data)
    }

    func listGPUTypes(token: String) async throws -> [RunPodGPUType] {
        let body = try JSONEncoder().encode(RunPodGraphQLRequest(query: Self.gpuTypesQuery))
        let request = try makeGraphQLRequest(token: token, body: body)
        let (data, response) = try await transport.data(for: request)
        try validate(response: response, data: data)

        let payload = try JSONDecoder().decode(RunPodGraphQLResponse<RunPodGPUTypesPayload>.self, from: data)
        if let errors = payload.errors, !errors.isEmpty {
            throw RunPodError.graphQLErrors(errors.map(\.message))
        }
        guard let data = payload.data else {
            throw RunPodError.invalidResponse
        }

        return data.gpuTypes
            .filter { $0.secureCloud != false }
            .sorted {
                $0.displayNameValue.localizedStandardCompare($1.displayNameValue) == .orderedAscending
            }
    }

    func listOfficialPodTemplates(token: String) async throws -> [RunPodTemplate] {
        let request = try makeRequest(
            path: "templates",
            method: "GET",
            token: token,
            queryItems: [
                URLQueryItem(name: "includeRunpodTemplates", value: "true"),
                URLQueryItem(name: "includePublicTemplates", value: "false"),
                URLQueryItem(name: "includeEndpointBoundTemplates", value: "false"),
            ]
        )
        let (data, response) = try await transport.data(for: request)
        try validate(response: response, data: data)
        return try JSONDecoder().decode([RunPodTemplate].self, from: data)
            .filter(\.isOfficialPodTemplate)
            .sorted {
                $0.displayNameValue.localizedStandardCompare($1.displayNameValue) == .orderedAscending
            }
    }

    func createPod(token: String, configuration: RunPodLaunchConfiguration) async throws -> RunPodPodSummary {
        let body = try JSONEncoder().encode(configuration.requestBody())
        let request = try makeRequest(path: "pods", method: "POST", token: token, body: body)
        let (data, response) = try await transport.data(for: request)
        try validate(response: response, data: data)
        return try JSONDecoder().decode(RunPodPodSummary.self, from: data)
    }

    func deletePod(token: String, podID: String) async throws {
        let encodedID = podID.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? podID
        let request = try makeRequest(path: "pods/\(encodedID)", method: "DELETE", token: token)
        let (data, response) = try await transport.data(for: request)
        try validate(response: response, data: data)
    }

    private func makeRequest(
        path: String,
        method: String,
        token: String,
        queryItems: [URLQueryItem] = [],
        body: Data? = nil
    ) throws -> URLRequest {
        let trimmedToken = token.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedToken.isEmpty else {
            throw RunPodError.missingAPIKey
        }

        let base = baseURL.appendingPathComponent(path)
        var components = URLComponents(url: base, resolvingAgainstBaseURL: false)
        if !queryItems.isEmpty {
            components?.queryItems = queryItems
        }
        guard let url = components?.url else {
            throw RunPodError.invalidResponse
        }
        var request = URLRequest(url: url)
        request.httpMethod = method
        request.setValue("Bearer \(trimmedToken)", forHTTPHeaderField: "Authorization")
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        if let body {
            request.httpBody = body
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        }
        return request
    }

    private func makeGraphQLRequest(token: String, body: Data) throws -> URLRequest {
        let trimmedToken = token.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedToken.isEmpty else {
            throw RunPodError.missingAPIKey
        }

        var request = URLRequest(url: graphQLURL)
        request.httpMethod = "POST"
        request.httpBody = body
        request.setValue("Bearer \(trimmedToken)", forHTTPHeaderField: "Authorization")
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        return request
    }

    private func validate(response: HTTPURLResponse, data: Data) throws {
        guard (200..<300).contains(response.statusCode) else {
            let body = String(data: data, encoding: .utf8)?
                .trimmingCharacters(in: .whitespacesAndNewlines)
            throw RunPodError.httpStatus(response.statusCode, body)
        }
    }

    private static let gpuTypesQuery = """
    query PiMPAudioRunPodGPUTypes {
      gpuTypes {
        id
        displayName
        manufacturer
        memoryInGb
        secureCloud
        secureLowestPrice: lowestPrice(input: { gpuCount: 1, secureCloud: true }) {
          ...RunPodLowestPriceFields
        }
        securePrice
        secureSpotPrice
        maxGpuCount
        maxGpuCountSecureCloud
      }
    }

    fragment RunPodLowestPriceFields on LowestPrice {
      gpuName
      gpuTypeId
      minimumBidPrice
      uninterruptablePrice
      stockStatus
      availableGpuCounts
      maxGpuCount
      maxUnreservedGpuCount
      rentedCount
      totalCount
    }
    """
}

enum RunPodError: LocalizedError, Equatable {
    case missingAPIKey
    case invalidConfiguration(String)
    case invalidResponse
    case graphQLErrors([String])
    case httpStatus(Int, String?)

    var errorDescription: String? {
        switch self {
        case .missingAPIKey:
            return "RunPod API key is not configured."
        case .invalidConfiguration(let message):
            return message
        case .invalidResponse:
            return "RunPod returned a response PiMP Audio Studio could not read."
        case .graphQLErrors(let messages):
            return "RunPod rejected the GPU type request: \(Self.joinedDetails(messages))"
        case .httpStatus(let status, let body):
            return Self.message(forHTTPStatus: status, body: body)
        }
    }

    static func userFacingMessage(for error: Error) -> String {
        if let runPodError = error as? RunPodError {
            return runPodError.localizedDescription
        }

        if let urlError = error as? URLError {
            switch urlError.code {
            case .notConnectedToInternet,
                 .cannotFindHost,
                 .cannotConnectToHost,
                 .networkConnectionLost,
                 .dnsLookupFailed,
                 .timedOut:
                return "Could not reach RunPod. Check your connection and try again. (\(urlError.localizedDescription))"
            default:
                return "RunPod network request failed: \(urlError.localizedDescription)"
            }
        }

        return error.localizedDescription
    }

    private static func message(forHTTPStatus status: Int, body: String?) -> String {
        let detail = cleanDetail(body)
        switch status {
        case 401:
            return "RunPod rejected the API key. Check that it was copied correctly or generate a new key in RunPod.\(detail)"
        case 403:
            return "RunPod accepted the request but this API key is not authorized for that action. Check the key permissions or account status.\(detail)"
        case 404:
            return "RunPod could not find that resource. Refresh pods and try again.\(detail)"
        case 429:
            return "RunPod rate-limited the request. Wait a minute and try again.\(detail)"
        case 500...599:
            return "RunPod is having trouble right now (HTTP \(status)). Try again shortly.\(detail)"
        default:
            return "RunPod request failed with HTTP \(status).\(detail)"
        }
    }

    private static func cleanDetail(_ body: String?) -> String {
        guard let body, !body.isEmpty else { return "" }
        let trimmed = body.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return "" }
        let capped = trimmed.count > 240 ? "\(trimmed.prefix(240))..." : trimmed
        return " Detail: \(capped)"
    }

    private static func joinedDetails(_ messages: [String]) -> String {
        let clean = messages
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        guard !clean.isEmpty else { return "unknown error" }
        return clean.joined(separator: "; ")
    }
}

enum RunPodRemoteWorkerLocator {
    static func defaultDirectoryURL() -> URL {
        if let environmentPath = ProcessInfo.processInfo.environment["PIMP_REMOTE_WORKER_DIR"]?
            .trimmingCharacters(in: .whitespacesAndNewlines),
           !environmentPath.isEmpty {
            let environmentURL = URL(fileURLWithPath: NSString(string: environmentPath).expandingTildeInPath)
            if isWorkerDirectory(environmentURL) {
                return environmentURL
            }
        }

        if let bundledURL = Bundle.main.resourceURL?.appendingPathComponent("remote-worker", isDirectory: true),
           isWorkerDirectory(bundledURL) {
            return bundledURL
        }

        let sourceTreeURL = sourceTreeDirectoryURL()
        if isWorkerDirectory(sourceTreeURL) {
            return sourceTreeURL
        }

        let currentDirectoryURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent("remote-worker", isDirectory: true)
        if isWorkerDirectory(currentDirectoryURL) {
            return currentDirectoryURL
        }

        return sourceTreeURL
    }

    private static func sourceTreeDirectoryURL() -> URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("remote-worker", isDirectory: true)
    }

    private static func isWorkerDirectory(_ url: URL) -> Bool {
        FileManager.default.fileExists(
            atPath: url.appendingPathComponent("install.sh").path(percentEncoded: false)
        )
    }
}

private func shellQuoted(_ value: String) -> String {
    "'\(shellSingleQuotedContent(value))'"
}

private func shellPathQuoted(_ value: String) -> String {
    if value.hasPrefix("~/") {
        let suffix = String(value.dropFirst(2))
        if suffix.rangeOfCharacter(from: shellUnsafePathCharacters) == nil {
            return value
        }
        return "~/" + shellQuoted(suffix)
    }
    if value.rangeOfCharacter(from: shellUnsafePathCharacters) == nil {
        return value
    }
    return shellQuoted(value)
}

private func shellSingleQuotedContent(_ value: String) -> String {
    value.replacingOccurrences(of: "'", with: "'\\''")
}

private let shellUnsafePathCharacters = CharacterSet.whitespacesAndNewlines
    .union(CharacterSet(charactersIn: "'\"\\$`!()[]{};&|<>*?"))
