import SwiftUI

struct IntelligenceSettingsView: View {
    @EnvironmentObject private var modelManager: LLMModelManager
    @EnvironmentObject private var apiKeyStore: APIKeyStore
    @EnvironmentObject private var lyricWriterManager: LyricWriterManager

    @State private var newAPIKeys: [String: String] = [:]
    @State private var customModelRepoID: String = ""
    @State private var customModelError: String?

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

                    // Green checkmark = model is BOTH active AND fully on disk.
                    // Was previously gated only on activeModel ID, which
                    // could show the check while a re-download was in flight
                    // or after files were partially deleted.
                    if modelManager.activeModel?.id == spec.id
                        && modelManager.isModelDownloaded(spec) {
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

            Divider()

            HStack {
                TextField("HuggingFace repo (e.g. mlx-community/Model-4bit)", text: $customModelRepoID)
                    .textFieldStyle(.roundedBorder)

                Button("Add Model") {
                    customModelError = nil
                    Task {
                        do {
                            try await modelManager.addCustomModel(repoID: customModelRepoID)
                            customModelRepoID = ""
                        } catch {
                            customModelError = "Invalid repo: missing required files"
                        }
                    }
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .disabled(customModelRepoID.isEmpty)
            }

            if let error = customModelError {
                Label(error, systemImage: "exclamationmark.triangle")
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
