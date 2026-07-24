import Foundation
import PiMPAudioCore
import SwiftUI

@MainActor
final class LyricWriterManager: ObservableObject {
    @Published var isWriting: Bool = false
    @Published var activeBackend: String = "local"
    @Published var lastError: String?

    static let remoteBackendProviders = APIKeyStore.providers
        .filter { $0.id != APIKeyStore.huggingFaceProviderID }

    private var backends: [String: any LyricWriter] = [:]
    private var writeTask: Task<Void, Never>?
    private var activeWriteID: UUID?
    private let apiKeyStore: APIKeyStore
    private let providerAccessStore: ProviderAccessStore

    init(apiKeyStore: APIKeyStore, providerAccessStore: ProviderAccessStore) {
        self.apiKeyStore = apiKeyStore
        self.providerAccessStore = providerAccessStore
    }

    func updateLLM(_ llm: LLMActor?) {
        if let llm {
            backends["local"] = LocalLyricWriter(llm: llm)
        } else {
            backends.removeValue(forKey: "local")
        }
    }

    func refreshAPIBackends() {
        refreshRemoteBackends()
    }

    func refreshRemoteBackends() {
        for provider in Self.remoteBackendProviders {
            if let apiBackend = apiBackend(for: provider.id) {
                backends[provider.id] = apiBackend
                continue
            }

            if let cliBackend = cliBackend(for: provider) {
                backends[provider.id] = cliBackend
            } else {
                backends.removeValue(forKey: provider.id)
            }
        }
    }

    func availableBackends() -> [(id: String, displayName: String)] {
        backends.map { ($0.key, $0.value.displayName) }.sorted { $0.id < $1.id }
    }

    func backendRequirementMessage(for backendID: String) -> String? {
        guard backendID != "local", backends[backendID] == nil else {
            return nil
        }

        guard let provider = Self.remoteBackendProviders.first(where: { $0.id == backendID }) else {
            return nil
        }

        if apiKeyStore.hasKey(for: backendID) {
            return nil
        }

        // Some backends (e.g. Gemini) are API-key only — no Local CLI Access entry.
        if ProviderCLIDefinition.all.first(where: { $0.providerID == backendID }) == nil {
            return "Add a \(provider.displayName) API key"
        }

        let status = providerAccessStore.status(
            for: provider.id,
            displayName: provider.displayName
        )

        switch status.testState {
        case .running:
            return "Checking local CLI access"
        case .passed:
            return nil
        case .workspaceTrustNeeded:
            return "Trust the workspace for \(provider.displayName) or add an API key"
        case .blocked:
            return "Grant local CLI access for \(provider.displayName) or add an API key"
        case .failed:
            return "Sign in to \(provider.displayName) CLI or add an API key"
        case .idle:
            break
        }

        switch status.state {
        case .authenticated:
            return nil
        case .installed:
            return "Test \(provider.displayName) CLI access or add an API key"
        case .unauthenticated:
            return "Sign in to \(provider.displayName) CLI or add an API key"
        case .workspaceTrustNeeded:
            return "Trust the workspace for \(provider.displayName) or add an API key"
        case .blocked:
            return "Grant local CLI access for \(provider.displayName) or add an API key"
        case .failed:
            return "Check \(provider.displayName) CLI access or add an API key"
        case .missing:
            return "Install \(provider.displayName) CLI or add an API key"
        case .notChecked:
            return "Refresh local CLI access or add an API key"
        }
    }

    private func apiBackend(for providerID: String) -> (any LyricWriter)? {
        guard let key = apiKeyStore.apiKey(for: providerID) else {
            return nil
        }

        switch providerID {
        case "anthropic":
            return ClaudeLyricWriter(apiKey: key)
        case "openai":
            return OpenAILyricWriter(apiKey: key)
        case "google":
            return GeminiLyricWriter(apiKey: key)
        case "xai":
            return XAILyricWriter(apiKey: key)
        default:
            return nil
        }
    }

    private func cliBackend(
        for provider: (id: String, displayName: String, envVar: String)
    ) -> ProviderCLILyricWriter? {
        let status = providerAccessStore.status(
            for: provider.id,
            displayName: provider.displayName
        )
        guard status.isUsableForLyricWriting,
              let command = status.command,
              let definition = ProviderCLIDefinition.all.first(where: { $0.providerID == provider.id }) else {
            return nil
        }

        return ProviderCLILyricWriter(
            providerID: provider.id,
            displayName: provider.displayName,
            command: command,
            homeDirectoryMode: definition.homeDirectoryMode
        )
    }

    func writeLyrics(
        prompt: String,
        style: String,
        duration: TimeInterval,
        into binding: Binding<String>
    ) {
        guard let backend = backends[activeBackend] ?? backends["local"] else {
            lastError = LyricWriterManagerError.noAvailableBackend.localizedDescription
            return
        }
        let writeID = beginWrite()
        binding.wrappedValue = ""

        writeTask = Task { [weak self] in
            defer { self?.finishWrite(id: writeID) }
            let stream = backend.write(prompt: prompt, style: style, duration: duration)
            var rawLyrics = ""
            do {
                for try await token in stream {
                    guard !Task.isCancelled, self?.isCurrentWrite(writeID) == true else {
                        throw CancellationError()
                    }
                    rawLyrics += token
                    binding.wrappedValue = rawLyrics
                }
                guard !Task.isCancelled, self?.isCurrentWrite(writeID) == true else {
                    throw CancellationError()
                }
                let sanitized = LyricOutputSanitizer.sanitize(
                    rawLyrics,
                    targetDuration: duration
                )
                let finalLyrics = sanitized.isEmpty
                    ? rawLyrics.trimmingCharacters(in: .whitespacesAndNewlines)
                    : sanitized
                binding.wrappedValue = finalLyrics
            } catch is CancellationError {
                // User cancellation leaves whatever partial text was already streamed.
            } catch {
                guard self?.isCurrentWrite(writeID) == true else { return }
                self?.lastError = error.localizedDescription
            }
        }
    }

    func writeLyricsText(
        prompt: String,
        style: String,
        duration: TimeInterval
    ) async throws -> String {
        guard let backend = backends[activeBackend] ?? backends["local"] else {
            throw LyricWriterManagerError.noAvailableBackend
        }

        let writeID = beginWrite()
        defer { finishWrite(id: writeID) }

        let stream = backend.write(prompt: prompt, style: style, duration: duration)
        var rawLyrics = ""
        for try await token in stream {
            guard !Task.isCancelled, isCurrentWrite(writeID) else {
                throw CancellationError()
            }
            rawLyrics += token
        }

        guard !Task.isCancelled, isCurrentWrite(writeID) else {
            throw CancellationError()
        }

        let sanitized = LyricOutputSanitizer.sanitize(
            rawLyrics,
            targetDuration: duration
        )
        let finalLyrics = sanitized.isEmpty
            ? rawLyrics.trimmingCharacters(in: .whitespacesAndNewlines)
            : sanitized
        return finalLyrics
    }

    func cancel() {
        writeTask?.cancel()
        writeTask = nil
        activeWriteID = nil
        isWriting = false
    }

    private func beginWrite() -> UUID {
        writeTask?.cancel()
        let writeID = UUID()
        activeWriteID = writeID
        isWriting = true
        lastError = nil
        return writeID
    }

    private func isCurrentWrite(_ writeID: UUID) -> Bool {
        activeWriteID == writeID
    }

    private func finishWrite(id: UUID) {
        guard activeWriteID == id else { return }
        activeWriteID = nil
        writeTask = nil
        isWriting = false
    }
}

enum LyricWriterManagerError: LocalizedError {
    case noAvailableBackend

    var errorDescription: String? {
        switch self {
        case .noAvailableBackend:
            return "No lyric writer backend is available. Activate the local LLM or configure an API backend."
        }
    }
}

private extension ProviderCLIStatus {
    var isUsableForLyricWriting: Bool {
        if case .authenticated = state {
            return true
        }
        if case .passed = testState {
            return true
        }
        return false
    }
}
