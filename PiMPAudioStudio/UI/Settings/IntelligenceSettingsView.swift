import SwiftUI

struct IntelligenceSettingsView: View {
    @EnvironmentObject private var modelManager: LLMModelManager
    @EnvironmentObject private var apiKeyStore: APIKeyStore
    @EnvironmentObject private var providerAccessStore: ProviderAccessStore
    @EnvironmentObject private var lyricWriterManager: LyricWriterManager

    @State private var newAPIKeys: [String: String] = [:]
    @State private var customModelRepoID: String = ""
    @State private var customModelError: String?

    var body: some View {
        Form {
            modelSection
            lyricBackendSection
            localCLIAccessSection
            apiKeysSection
        }
        .formStyle(.grouped)
        .frame(minWidth: 500, minHeight: 400)
        .task {
            await providerAccessStore.refreshIfNeeded()
            lyricWriterManager.refreshRemoteBackends()
        }
        .onChange(of: apiKeyStore.availableProviders) { _, _ in
            lyricWriterManager.refreshRemoteBackends()
        }
        .onChange(of: providerAccessStore.statuses) { _, _ in
            lyricWriterManager.refreshRemoteBackends()
        }
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
                ForEach(LyricWriterManager.remoteBackendProviders, id: \.id) { provider in
                    Text(provider.displayName).tag(provider.id)
                }
            }
            .pickerStyle(.menu)

            if let message = lyricWriterManager.backendRequirementMessage(
                for: lyricWriterManager.activeBackend
            ) {
                Label(message, systemImage: "exclamationmark.triangle")
                    .font(.caption)
                    .foregroundStyle(.orange)
            }
        }
    }

    private var localCLIAccessSection: some View {
        Section {
            // Only providers with a registered CLI definition appear here (not API-only
            // backends such as Gemini, which is key-based for lyric writing).
            ForEach(ProviderCLIDefinition.all, id: \.providerID) { definition in
                let status = providerAccessStore.status(
                    for: definition.providerID,
                    displayName: definition.displayName
                )

                HStack(spacing: 12) {
                    Image(systemName: statusIcon(for: status))
                        .foregroundStyle(statusTint(for: status))
                        .frame(width: 18)

                    VStack(alignment: .leading, spacing: 3) {
                        Text(definition.displayName)
                            .font(.body)

                        Text(statusTitle(for: status))
                            .font(.caption)
                            .foregroundStyle(statusTint(for: status))

                        if let detail = statusDetail(for: status) {
                            Text(detail)
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                                .lineLimit(2)
                        }
                    }

                    Spacer()

                    if case .running = status.testState {
                        ProgressView()
                            .controlSize(.small)
                    }

                    if status.canGrantAccess {
                        Button {
                            Task {
                                await providerAccessStore.grantAccess(
                                    providerID: definition.providerID
                                )
                            }
                        } label: {
                            Image(systemName: "folder.badge.plus")
                        }
                        .buttonStyle(.borderless)
                        .help("Grant local CLI folder access")
                    }

                    Button {
                        Task {
                            await providerAccessStore.test(providerID: definition.providerID)
                        }
                    } label: {
                        Image(systemName: "checkmark.seal")
                    }
                    .buttonStyle(.borderless)
                    .disabled(!status.canTest || isTesting(status))
                    .help("Test local CLI access")
                }
            }
        } header: {
            HStack {
                Text("Local CLI Access")
                Spacer()

                if providerAccessStore.isRefreshing {
                    ProgressView()
                        .controlSize(.small)
                }

                Button {
                    Task { await providerAccessStore.refresh() }
                } label: {
                    Image(systemName: "arrow.clockwise")
                }
                .buttonStyle(.borderless)
                .disabled(providerAccessStore.isRefreshing)
                .help("Refresh local CLI access")
            }
        }
    }

    private var apiKeysSection: some View {
        Section("API Keys (Optional)") {
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

    private func statusTitle(for status: ProviderCLIStatus) -> String {
        switch status.testState {
        case .running:
            return "Testing..."
        case .passed:
            return "Test passed"
        case .workspaceTrustNeeded:
            return "Workspace trust needed"
        case .blocked:
            return "Test blocked"
        case .failed:
            return "Test failed"
        case .idle:
            break
        }

        switch status.state {
        case .notChecked:
            return "Not checked"
        case .missing:
            return "CLI not found"
        case .installed:
            return "Installed"
        case .authenticated:
            return "Signed in"
        case .unauthenticated:
            return "Sign-in needed"
        case .workspaceTrustNeeded:
            return "Workspace trust needed"
        case .blocked:
            return "Sandbox access needed"
        case .failed:
            return "Check failed"
        }
    }

    private func statusDetail(for status: ProviderCLIStatus) -> String? {
        switch status.testState {
        case .passed(let detail), .workspaceTrustNeeded(let detail), .blocked(let detail), .failed(let detail):
            return detail
        case .running, .idle:
            break
        }

        var details: [String] = []
        if let command = status.command {
            details.append(command)
        }
        if let version = status.version {
            details.append(version)
        }
        if let detail = status.detail {
            details.append(detail)
        }
        return details.isEmpty ? nil : details.joined(separator: " - ")
    }

    private func statusIcon(for status: ProviderCLIStatus) -> String {
        switch status.testState {
        case .running:
            return "clock"
        case .passed:
            return "checkmark.circle.fill"
        case .workspaceTrustNeeded:
            return "folder.badge.questionmark"
        case .blocked:
            return "lock.fill"
        case .failed:
            return "xmark.circle.fill"
        case .idle:
            break
        }

        switch status.state {
        case .authenticated:
            return "checkmark.circle.fill"
        case .installed:
            return "terminal"
        case .unauthenticated:
            return "person.crop.circle.badge.exclamationmark"
        case .workspaceTrustNeeded:
            return "folder.badge.questionmark"
        case .blocked:
            return "lock.fill"
        case .missing, .notChecked:
            return "circle"
        case .failed:
            return "exclamationmark.triangle.fill"
        }
    }

    private func statusTint(for status: ProviderCLIStatus) -> Color {
        switch status.testState {
        case .running:
            return .blue
        case .passed:
            return .green
        case .workspaceTrustNeeded:
            return .orange
        case .blocked:
            return .orange
        case .failed:
            return .red
        case .idle:
            break
        }

        switch status.state {
        case .authenticated:
            return .green
        case .installed:
            return .blue
        case .unauthenticated:
            return .orange
        case .workspaceTrustNeeded:
            return .orange
        case .blocked:
            return .orange
        case .failed:
            return .red
        case .missing, .notChecked:
            return .secondary
        }
    }

    private func isTesting(_ status: ProviderCLIStatus) -> Bool {
        if case .running = status.testState {
            return true
        }
        return false
    }
}
