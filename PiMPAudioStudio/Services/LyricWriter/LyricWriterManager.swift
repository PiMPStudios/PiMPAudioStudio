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
        writeTask?.cancel()
        isWriting = true
        lastError = nil
        binding.wrappedValue = ""

        writeTask = Task { [weak self] in
            let stream = backend.write(prompt: prompt, style: style, duration: duration)
            var rawLyrics = ""
            do {
                for try await token in stream {
                    if Task.isCancelled { break }
                    rawLyrics += token
                    binding.wrappedValue = rawLyrics
                }
                if !Task.isCancelled {
                    let sanitized = LyricOutputSanitizer.sanitize(
                        rawLyrics,
                        targetDuration: duration
                    )
                    let finalLyrics = sanitized.isEmpty
                        ? rawLyrics.trimmingCharacters(in: .whitespacesAndNewlines)
                        : sanitized
                    binding.wrappedValue = finalLyrics
                }
            } catch is CancellationError {
                // User cancellation leaves whatever partial text was already streamed.
            } catch {
                self?.lastError = error.localizedDescription
            }
            guard let self else { return }
            self.isWriting = false
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

        writeTask?.cancel()
        isWriting = true
        lastError = nil
        defer { isWriting = false }

        let stream = backend.write(prompt: prompt, style: style, duration: duration)
        var rawLyrics = ""
        for try await token in stream {
            if Task.isCancelled {
                throw CancellationError()
            }
            rawLyrics += token
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
