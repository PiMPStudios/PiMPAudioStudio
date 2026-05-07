import SwiftUI
import AnvilCore

struct ModelsSettingsView: View {
    @EnvironmentObject private var appState: AppState
    @EnvironmentObject private var generationService: GenerationService
    @EnvironmentObject private var sfxModelManager: SFXModelManager

    @State private var aceStepSpecs: [ModelSpec] = []
    @State private var stableAudioSpecs: [ModelSpec] = []
    @State private var activeModels: [ModelCategory: String] = [:]
    @State private var statuses: [String: ModelStatus] = [:]
    @State private var showDeleteConfirmation: String?
    @State private var showRemoveConfirmation: String?
    @State private var errorMessage: String?
    @State private var refreshTask: Task<Void, Never>?

    var body: some View {
        Form {
            aceStepSection
            stableAudioSection
            memorySection
            AddModelView(
                registry: appState.registry,
                onModelAdded: { refreshModels() }
            )
        }
        .formStyle(.grouped)
        .frame(minWidth: 500, minHeight: 400)
        .task { refreshModels() }
        .confirmationDialog(
            "Delete model files?",
            isPresented: Binding(
                get: { showDeleteConfirmation != nil },
                set: { if !$0 { showDeleteConfirmation = nil } }
            )
        ) {
            if let id = showDeleteConfirmation {
                Button("Delete Files", role: .destructive) {
                    deleteFiles(id: id)
                }
            }
        } message: {
            Text("You can re-download later.")
        }
        .confirmationDialog(
            "Remove model?",
            isPresented: Binding(
                get: { showRemoveConfirmation != nil },
                set: { if !$0 { showRemoveConfirmation = nil } }
            )
        ) {
            if let id = showRemoveConfirmation {
                Button("Remove", role: .destructive) {
                    removeModel(id: id)
                }
            }
        } message: {
            Text("This will delete all downloaded files and remove the model from your list.")
        }
        .alert("Error", isPresented: Binding(
            get: { errorMessage != nil },
            set: { if !$0 { errorMessage = nil } }
        )) {
            Button("OK", role: .cancel) { }
        } message: {
            if let msg = errorMessage {
                Text(msg)
            }
        }
    }

    private var aceStepSection: some View {
        Section("ACE-Step (Music)") {
            ForEach(aceStepSpecs) { spec in
                ModelRow(
                    spec: spec,
                    isActive: activeModels[spec.category] == spec.id,
                    status: statuses[spec.id] ?? .notDownloaded,
                    isBuiltIn: appState.registry.isBuiltIn(id: spec.id),
                    onSetActive: { setActive(spec) },
                    onDownload: { download(spec) },
                    onDelete: { showDeleteConfirmation = spec.id },
                    onRemove: { showRemoveConfirmation = spec.id }
                )
            }
        }
    }

    private var stableAudioSection: some View {
        Section("Stable Audio (SFX)") {
            ForEach(stableAudioSpecs) { spec in
                ModelRow(
                    spec: spec,
                    isActive: activeModels[spec.category] == spec.id,
                    status: statuses[spec.id] ?? .notDownloaded,
                    isBuiltIn: appState.registry.isBuiltIn(id: spec.id),
                    onSetActive: { setActive(spec) },
                    onDownload: { download(spec) },
                    onDelete: { showDeleteConfirmation = spec.id },
                    onRemove: { showRemoveConfirmation = spec.id }
                )
            }
        }
    }

    private var memorySection: some View {
        Section("Memory") {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Release GPU Memory")
                        .font(.body)
                    Text("Unloads all model weights. They reload automatically on next use.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Button("Release") {
                    Task {
                        await appState.unloadAllModels(
                            generationService: generationService,
                            sfxModelManager: sfxModelManager
                        )
                    }
                }
                .buttonStyle(.bordered)
            }
        }
    }

    private func refreshModels() {
        refreshTask?.cancel()
        refreshTask = Task { @MainActor in
            let registry = appState.registry
            aceStepSpecs = await registry.specs(for: .aceStepDiT)
                + await registry.specs(for: .aceStepVAE)
            stableAudioSpecs = await registry.specs(for: .stableAudio)

            var newStatuses: [String: ModelStatus] = [:]
            for spec in aceStepSpecs + stableAudioSpecs {
                newStatuses[spec.id] = await registry.status(for: spec.id)
            }
            statuses = newStatuses

            activeModels = [:]
            for cat: ModelCategory in [.aceStepDiT, .aceStepVAE, .stableAudio] {
                if let id = await registry.activeModelID(for: cat) {
                    activeModels[cat] = id
                }
            }
        }
    }

    private func setActive(_ spec: ModelSpec) {
        Task {
            do {
                try await appState.registry.setActiveModel(id: spec.id, for: spec.category)
            } catch {
                errorMessage = error.localizedDescription
            }
            refreshModels()
        }
    }

    private func download(_ spec: ModelSpec) {
        Task {
            do {
                try await appState.registry.downloadModels([spec]) { _, progress in
                    Task { @MainActor in
                        statuses[spec.id] = .downloading(progress: progress)
                    }
                }
            } catch {
                errorMessage = error.localizedDescription
            }
            refreshModels()
        }
    }

    private func deleteFiles(id: String) {
        Task {
            do {
                try await appState.registry.deleteModelFiles(id: id)
            } catch {
                errorMessage = error.localizedDescription
            }
            refreshModels()
        }
    }

    private func removeModel(id: String) {
        Task {
            do {
                try await appState.registry.deleteModelFiles(id: id)
                try await appState.registry.removeModel(id: id)
            } catch {
                errorMessage = error.localizedDescription
            }
            refreshModels()
        }
    }
}

// MARK: - Model Row

private struct ModelRow: View {
    let spec: ModelSpec
    let isActive: Bool
    let status: ModelStatus
    let isBuiltIn: Bool
    let onSetActive: () -> Void
    let onDownload: () -> Void
    let onDelete: () -> Void
    let onRemove: () -> Void

    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 6) {
                    Text(spec.displayName)
                        .font(.body)
                        .fontWeight(isActive ? .semibold : .regular)
                    if isActive {
                        Image(systemName: "checkmark.circle.fill")
                            .foregroundStyle(.accent)
                            .font(.caption)
                    }
                }
                Text(spec.totalSizeFormatted)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Spacer()

            statusBadge

            actions
        }
        .padding(.vertical, 2)
        .listRowBackground(isActive ? Color.accentColor.opacity(0.08) : nil)
    }

    @ViewBuilder
    private var statusBadge: some View {
        switch status {
        case .ready:
            Text("Ready")
                .font(.caption2)
                .padding(.horizontal, 6)
                .padding(.vertical, 2)
                .background(.green.opacity(0.15), in: Capsule())
                .foregroundStyle(.green)
        case .notDownloaded:
            Text("Not Downloaded")
                .font(.caption2)
                .padding(.horizontal, 6)
                .padding(.vertical, 2)
                .background(.secondary.opacity(0.15), in: Capsule())
                .foregroundStyle(.secondary)
        case .downloading(let progress):
            HStack(spacing: 4) {
                ProgressView(value: progress)
                    .frame(width: 40)
                Text("Downloading \(Int(progress * 100))%")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        case .verifying:
            Text("Verifying...")
                .font(.caption2)
                .foregroundStyle(.secondary)
        case .failed(let message):
            Text(message)
                .font(.caption2)
                .foregroundStyle(.red)
                .lineLimit(1)
        }
    }

    @ViewBuilder
    private var actions: some View {
        switch status {
        case .ready:
            HStack(spacing: 4) {
                if !isActive {
                    Button("Set Active") { onSetActive() }
                        .controlSize(.small)
                }
                Button(role: .destructive) { onDelete() } label: {
                    Image(systemName: "trash")
                }
                .controlSize(.small)
                if !isBuiltIn {
                    Button(role: .destructive) { onRemove() } label: {
                        Image(systemName: "xmark.circle")
                    }
                    .controlSize(.small)
                    .help("Remove model from list")
                }
            }
        case .notDownloaded:
            HStack(spacing: 4) {
                Button("Download") { onDownload() }
                    .controlSize(.small)
                if !isBuiltIn {
                    Button(role: .destructive) { onRemove() } label: {
                        Image(systemName: "xmark.circle")
                    }
                    .controlSize(.small)
                    .help("Remove model from list")
                }
            }
        case .failed:
            Button("Retry") { onDownload() }
                .controlSize(.small)
        default:
            EmptyView()
        }
    }
}
