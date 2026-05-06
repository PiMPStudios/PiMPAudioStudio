import SwiftUI

struct IntelligenceSettingsView: View {
    @EnvironmentObject private var modelManager: LLMModelManager
    @EnvironmentObject private var apiKeyStore: APIKeyStore
    @EnvironmentObject private var lyricWriterManager: LyricWriterManager

    @State private var newAPIKeys: [String: String] = [:]

    var body: some View {
        Form {
            modelSection
            lyricBackendSection
            apiKeysSection
        }
        .formStyle(.grouped)
        .frame(minWidth: 500, minHeight: 400)
    }

    private var modelSection: some View {
        Section("Local Model") {
            ForEach(modelManager.availableModels) { spec in
                HStack {
                    VStack(alignment: .leading) {
                        Text(spec.displayName)
                            .font(.body)
                        Text(ByteCountFormatter.string(fromByteCount: spec.sizeBytes, countStyle: .file))
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }

                    Spacer()

                    if modelManager.activeModel?.id == spec.id {
                        Image(systemName: "checkmark.circle.fill")
                            .foregroundStyle(.green)
                    }

                    if modelManager.isModelDownloaded(spec) {
                        if modelManager.activeModel?.id != spec.id {
                            Button("Activate") {
                                modelManager.setActiveModel(spec)
                            }
                            .buttonStyle(.bordered)
                            .controlSize(.small)
                        }
                    } else {
                        Button("Download") {
                            Task { try? await modelManager.downloadModel(spec) }
                        }
                        .buttonStyle(.borderedProminent)
                        .controlSize(.small)
                    }
                }
            }

            if case .downloading(let progress) = modelManager.downloadState {
                ProgressView("Downloading model...", value: Double(progress), total: 1.0)
            }

            if case .error(let message) = modelManager.downloadState {
                Label(message, systemImage: "exclamationmark.triangle")
                    .foregroundStyle(.red)
                    .font(.caption)
            }
        }
    }

    private var lyricBackendSection: some View {
        Section("Lyric Writer") {
            Picker("Backend", selection: $lyricWriterManager.activeBackend) {
                Text("Local (On-Device)").tag("local")
                ForEach(APIKeyStore.providers, id: \.id) { provider in
                    Text(provider.displayName).tag(provider.id)
                }
            }
            .pickerStyle(.menu)

            if lyricWriterManager.activeBackend != "local"
                && !apiKeyStore.hasKey(for: lyricWriterManager.activeBackend) {
                Label("API key required for this backend", systemImage: "exclamationmark.triangle")
                    .font(.caption)
                    .foregroundStyle(.orange)
            }
        }
    }

    private var apiKeysSection: some View {
        Section("API Keys") {
            ForEach(APIKeyStore.providers, id: \.id) { provider in
                HStack {
                    VStack(alignment: .leading) {
                        Text(provider.displayName)
                            .font(.body)
                        switch apiKeyStore.keySource(for: provider.id) {
                        case .keychain:
                            Text("Configured")
                                .font(.caption)
                                .foregroundStyle(.green)
                        case .environment:
                            Text("From environment (\(provider.envVar))")
                                .font(.caption)
                                .foregroundStyle(.blue)
                        case .none:
                            Text("Not configured")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }

                    Spacer()

                    SecureField("API Key", text: Binding(
                        get: { newAPIKeys[provider.id] ?? "" },
                        set: { newAPIKeys[provider.id] = $0 }
                    ))
                    .frame(width: 200)
                    .onSubmit {
                        if let key = newAPIKeys[provider.id], !key.isEmpty {
                            apiKeyStore.setAPIKey(key, for: provider.id)
                            newAPIKeys[provider.id] = ""
                        }
                    }

                    if apiKeyStore.keySource(for: provider.id) == .keychain {
                        Button(role: .destructive) {
                            apiKeyStore.removeAPIKey(for: provider.id)
                        } label: {
                            Image(systemName: "trash")
                        }
                        .buttonStyle(.borderless)
                    }
                }
            }
        }
    }
}
