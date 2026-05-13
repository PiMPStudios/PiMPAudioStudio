import Foundation
import AnvilCore
import SwiftUI

@MainActor
final class LyricWriterManager: ObservableObject {
    @Published var isWriting: Bool = false
    @Published var activeBackend: String = "local"

    private var backends: [String: any LyricWriter] = [:]
    private var writeTask: Task<Void, Never>?
    private let apiKeyStore: APIKeyStore

    init(apiKeyStore: APIKeyStore) {
        self.apiKeyStore = apiKeyStore
    }

    func updateLLM(_ llm: LLMActor?) {
        if let llm {
            backends["local"] = LocalLyricWriter(llm: llm)
        } else {
            backends.removeValue(forKey: "local")
        }
    }

    func refreshAPIBackends() {
        if let key = apiKeyStore.apiKey(for: "anthropic") {
            backends["anthropic"] = ClaudeLyricWriter(apiKey: key)
        } else {
            backends.removeValue(forKey: "anthropic")
        }
        if let key = apiKeyStore.apiKey(for: "openai") {
            backends["openai"] = OpenAILyricWriter(apiKey: key)
        } else {
            backends.removeValue(forKey: "openai")
        }
        if let key = apiKeyStore.apiKey(for: "google") {
            backends["google"] = GeminiLyricWriter(apiKey: key)
        } else {
            backends.removeValue(forKey: "google")
        }
        if let key = apiKeyStore.apiKey(for: "xai") {
            backends["xai"] = XAILyricWriter(apiKey: key)
        } else {
            backends.removeValue(forKey: "xai")
        }
    }

    func availableBackends() -> [(id: String, displayName: String)] {
        backends.map { ($0.key, $0.value.displayName) }.sorted { $0.id < $1.id }
    }

    func writeLyrics(
        prompt: String,
        style: String,
        duration: TimeInterval,
        into binding: Binding<String>
    ) {
        guard let backend = backends[activeBackend] ?? backends["local"] else { return }
        writeTask?.cancel()
        isWriting = true
        binding.wrappedValue = ""

        writeTask = Task { [weak self] in
            let stream = backend.write(prompt: prompt, style: style, duration: duration)
            var rawLyrics = ""
            for await token in stream {
                if Task.isCancelled { break }
                rawLyrics += token
                binding.wrappedValue = rawLyrics
            }
            if !Task.isCancelled {
                let sanitized = LyricOutputSanitizer.sanitize(rawLyrics)
                let finalLyrics = sanitized.isEmpty
                    ? rawLyrics.trimmingCharacters(in: .whitespacesAndNewlines)
                    : sanitized
                binding.wrappedValue = finalLyrics
            }
            guard let self else { return }
            self.isWriting = false
        }
    }

    func cancel() {
        writeTask?.cancel()
        isWriting = false
    }
}
