import SwiftUI
import PiMPAudioCore

struct ModelsSettingsView: View {
    @EnvironmentObject private var appState: AppState
    @EnvironmentObject private var generationService: GenerationService
    @EnvironmentObject private var apiKeyStore: APIKeyStore

    @State private var huggingFaceToken: String = ""
    @State private var aceStepSpecs: [ModelSpec] = []
    @State private var aceStepLMSpecs: [ModelSpec] = []
    @State private var stableAudioSpecs: [ModelSpec] = []
    @State private var activeModels: [ModelCategory: String] = [:]
    @State private var statuses: [String: ModelStatus] = [:]
    @State private var showDeleteConfirmation: String?
    @State private var showRemoveConfirmation: String?
    @State private var showNativePackDeleteConfirmation:
        DatasetAnnotationTask?
    @State private var errorMessage: String?
    @State private var refreshTask: Task<Void, Never>?
    @State private var isReleasingMemory: Bool = false
    @State private var nativeAnnotationAvailability:
        DatasetAnnotationBackendAvailability?
    @State private var nativePackInstallProgress:
        [DatasetAnnotationTask: NativeModelPackInstallProgress] = [:]
    @State private var nativePackInstallTasks:
        [DatasetAnnotationTask: Task<Void, Never>] = [:]
    @State private var nativePackCancellationRequests:
        Set<DatasetAnnotationTask> = []

    var body: some View {
        Form {
            huggingFaceTokenSection
            aceStepSection
            aceStepLMSection
            nativeAnnotationSection
            stableAudioSection
            memorySection
            AddModelView(
                registry: appState.registry,
                onModelAdded: {
                    refreshModels()
                    postModelRegistryChanged()
                },
                onModelStatusChanged: { id, status in
                    statuses[id] = status
                }
            )
        }
        .formStyle(.grouped)
        .frame(minWidth: 500, minHeight: 400)
        .task {
            huggingFaceToken = apiKeyStore.huggingFaceToken() ?? ""
            refreshModels()
        }
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
        .confirmationDialog(
            "Delete Native ACE model files?",
            isPresented: Binding(
                get: { showNativePackDeleteConfirmation != nil },
                set: {
                    if !$0 {
                        showNativePackDeleteConfirmation = nil
                    }
                }
            )
        ) {
            if let task = showNativePackDeleteConfirmation {
                Button("Delete Files", role: .destructive) {
                    deleteNativeAnnotationPack(task: task)
                }
            }
        } message: {
            Text("The model will become unavailable until you download it again.")
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

    private var huggingFaceTokenSection: some View {
        Section("HuggingFace Access Token (optional)") {
            SecureField("hf_…", text: $huggingFaceToken)
            HStack {
                Button("Save") {
                    do {
                        try apiKeyStore.setAPIKey(huggingFaceToken, for: APIKeyStore.huggingFaceProviderID)
                    } catch {
                        errorMessage = error.localizedDescription
                    }
                }
                .disabled(huggingFaceToken.isEmpty)
                Spacer()
                Link("Get a token", destination: URL(string: "https://huggingface.co/settings/tokens")!)
            }
            Text("Required for gated models like Stable Audio Open. Stored in macOS Keychain.")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    private var aceStepSection: some View {
        Section("ACE-Step (Music)") {
            let ditSpecs = aceStepSpecs.filter { $0.category == .aceStepDiT }
            let vaeSpecs = aceStepSpecs.filter { $0.category == .aceStepVAE }
            let ditByFamily = Dictionary(grouping: ditSpecs) { $0.variantFamily ?? "ungrouped" }

            Text("Pick one DiT quality/speed tier. All tiers share the VAE + Text Encoder below.")
                .font(.caption)
                .foregroundStyle(.secondary)

            ForEach(ditByFamily.keys.sorted(by: familySort), id: \.self) { family in
                let variants = (ditByFamily[family] ?? []).sorted(by: modelTierSort)
                if variants.count > 1 {
                    Text(familyDisplayName(family))
                        .font(.headline)
                        .padding(.top, 4)
                    ForEach(variants, id: \.id) { spec in
                        modelRow(for: spec)
                            .padding(.leading, 12)
                    }
                } else if let only = variants.first {
                    modelRow(for: only)
                }
            }

            if !vaeSpecs.isEmpty {
                Text("Shared Components")
                    .font(.headline)
                    .padding(.top, 6)
                Text("Required once for every ACE-Step music tier. This is the 48 kHz VAE decoder plus Qwen text encoder/tokenizer, not a separate quality choice.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            ForEach(vaeSpecs, id: \.id) { spec in
                modelRow(for: spec)
                    .padding(.leading, 12)
            }
        }
    }

    private func modelRow(for spec: ModelSpec) -> some View {
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

    private func familyDisplayName(_ family: String) -> String {
        switch family {
        case "ace-step-1.5": return "ACE-Step v1.5"
        case "ace-step-1.5-xl": return "ACE-Step v1.5 XL"
        case "ungrouped":    return ""
        default:             return family.replacingOccurrences(of: "-", with: " ").capitalized
        }
    }

    private func familySort(_ lhs: String, _ rhs: String) -> Bool {
        func rank(_ family: String) -> Int {
            switch family {
            case "ace-step-1.5": return 0
            case "ace-step-1.5-xl": return 1
            case "ungrouped": return 99
            default: return 50
            }
        }

        let lhsRank = rank(lhs)
        let rhsRank = rank(rhs)
        if lhsRank != rhsRank {
            return lhsRank < rhsRank
        }
        return lhs.localizedStandardCompare(rhs) == .orderedAscending
    }

    private func modelTierSort(_ lhs: ModelSpec, _ rhs: ModelSpec) -> Bool {
        let lhsRank = modelTierRank(lhs.id)
        let rhsRank = modelTierRank(rhs.id)
        if lhsRank != rhsRank {
            return lhsRank < rhsRank
        }
        return lhs.displayName.localizedStandardCompare(rhs.displayName) == .orderedAscending
    }

    private func modelTierRank(_ id: String) -> Int {
        if id.contains("turbo") { return 0 }
        if id.contains("base") { return 1 }
        if id.contains("sft") { return 2 }
        return 50
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

    private var nativeAnnotationSection: some View {
        Section("Dataset Annotation (Advanced)") {
            Text(
                "Download optional four-bit ACE captioner and transcriber "
                    + "packs for whole-track annotation with Swift and MLX. "
                    + "Every hosted file is checksum-verified before install. "
                    + "Automatic enrichment falls back to the lightweight "
                    + "native backend when these packs are unavailable."
            )
            .font(.caption)
            .foregroundStyle(.secondary)

            HStack(spacing: 4) {
                Text("Hosted packs:")
                    .foregroundStyle(.secondary)
                Link(
                    "Captioner",
                    destination:
                        NativeModelPackCatalog.captioner.modelPageURL
                )
                Text("·")
                    .foregroundStyle(.secondary)
                Link(
                    "Transcriber",
                    destination:
                        NativeModelPackCatalog.transcriber.modelPageURL
                )
            }
            .font(.caption2)

            if let availability = nativeAnnotationAvailability {
                HStack {
                    Label(
                        hardwareTierTitle(availability.hardwareTier),
                        systemImage: hardwareTierSystemImage(
                            availability.hardwareTier
                        )
                    )
                    .foregroundStyle(
                        availability.hardwareTier == .unsupported
                            ? .orange
                            : .secondary
                    )
                    Spacer()
                    Text(memorySummary)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                ForEach(availability.models, id: \.task) { model in
                    let catalogEntry = NativeModelPackCatalog.entry(
                        for: model.task
                    )
                    HStack {
                        VStack(alignment: .leading, spacing: 2) {
                            Text(catalogEntry.displayName)
                                .font(.body)
                            Text(
                                formattedByteCount(
                                    catalogEntry.expectedSizeBytes
                                )
                            )
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        }
                        Spacer()
                        if let progress =
                            nativePackInstallProgress[model.task] {
                            nativePackProgressView(
                                progress,
                                task: model.task
                            )
                        } else if model.isInstalled {
                            nativePackStatusBadge(
                                "Ready",
                                color: .green
                            )
                            Button(role: .destructive) {
                                showNativePackDeleteConfirmation =
                                    model.task
                            } label: {
                                Image(systemName: "trash")
                            }
                            .controlSize(.small)
                            .disabled(isNativePackOperationActive)
                        } else {
                            nativePackStatusBadge(
                                "Not Downloaded",
                                color: .secondary
                            )
                            Button("Download") {
                                downloadNativeAnnotationPack(
                                    task: model.task
                                )
                            }
                            .controlSize(.small)
                            .disabled(isNativePackOperationActive)
                        }
                    }
                    .padding(.vertical, 2)

                    if !model.isInstalled,
                       nativePackInstallProgress[model.task] == nil,
                       FileManager.default.fileExists(
                           atPath: model.directory.path
                       ),
                       let reason = model.unavailableReason {
                        Text(reason)
                            .font(.caption2)
                            .foregroundStyle(.red)
                            .lineLimit(2)
                    }
                }
            } else {
                ProgressView()
                    .controlSize(.small)
            }

            Button("Reveal Model Folder") {
                revealNativeAnnotationModelFolder()
            }
            .buttonStyle(.bordered)

            Text(
                "Measured policy: 16 GB is unsupported, 24 GB is provisional, "
                    + "and 32 GB or more is recommended. Both packs use about "
                    + "9.34 GB of disk."
            )
            .font(.caption)
            .foregroundStyle(.secondary)
        }
    }

    private var isNativePackOperationActive: Bool {
        !nativePackInstallTasks.isEmpty
    }

    private func nativePackStatusBadge(
        _ title: String,
        color: Color
    ) -> some View {
        Text(title)
            .font(.caption2)
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(color.opacity(0.15), in: Capsule())
            .foregroundStyle(color)
    }

    @ViewBuilder
    private func nativePackProgressView(
        _ progress: NativeModelPackInstallProgress,
        task: DatasetAnnotationTask
    ) -> some View {
        VStack(alignment: .trailing, spacing: 2) {
            ProgressView(value: progress.fractionCompleted)
                .frame(width: 76)
            Text(nativePackProgressLabel(progress))
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
        .help(progress.currentFile ?? nativePackProgressLabel(progress))

        if progress.stage != .installing
            && progress.stage != .complete {
            Button {
                cancelNativeAnnotationPackDownload(task: task)
            } label: {
                Image(systemName: "xmark.circle")
            }
            .buttonStyle(.borderless)
            .help("Cancel download")
            .disabled(nativePackCancellationRequests.contains(task))
        }
    }

    private func nativePackProgressLabel(
        _ progress: NativeModelPackInstallProgress
    ) -> String {
        switch progress.stage {
        case .preparing:
            return "Preparing…"
        case .downloading:
            return "Downloading \(Int(progress.fractionCompleted * 100))%"
        case .verifying:
            return "Verifying…"
        case .installing:
            return "Installing…"
        case .complete:
            return "Installed"
        }
    }

    private func formattedByteCount(_ bytes: Int64) -> String {
        ByteCountFormatter.string(
            fromByteCount: bytes,
            countStyle: .file
        )
    }

    private var aceStepLMSection: some View {
        Section {
            VStack(alignment: .leading, spacing: 4) {
                Text("Used for cover-song generation (audio→audio). Pick one variant — bigger is higher quality but slower.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            ForEach(aceStepLMSpecs.sorted(by: { $0.id < $1.id })) { spec in
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
        } header: {
            Text("ACE-Step LM (audio codes)")
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
                    releaseMemory()
                }
                .buttonStyle(.bordered)
                .disabled(isReleasingMemory)
                if isReleasingMemory {
                    ProgressView()
                        .controlSize(.small)
                }
            }
        }
    }

    private func releaseMemory() {
        guard !isReleasingMemory else { return }
        isReleasingMemory = true
        Task {
            _ = await generationService.unloadAllWeights()
            await MainActor.run {
                isReleasingMemory = false
            }
        }
    }

    private func refreshModels() {
        refreshTask?.cancel()
        refreshTask = Task { @MainActor in
            let registry = appState.registry
            let ditSpecs = await registry.specs(for: .aceStepDiT)
            let vaeSpecs = await registry.specs(for: .aceStepVAE)
            aceStepSpecs = ditSpecs + vaeSpecs
            aceStepLMSpecs = await registry.specs(for: .aceStepLM)
            stableAudioSpecs = await registry.specs(for: .stableAudio)
            nativeAnnotationAvailability =
                await NativeACEDatasetAnnotationBackend(
                    modelsDirectory: AppStorageLocations.modelsDirectory
                ).availability()

            var newStatuses: [String: ModelStatus] = [:]
            for spec in aceStepSpecs + aceStepLMSpecs + stableAudioSpecs {
                newStatuses[spec.id] = await registry.status(for: spec.id)
            }
            statuses = newStatuses

            activeModels = [:]
            for cat: ModelCategory in [.aceStepDiT, .aceStepVAE, .aceStepLM, .stableAudio] {
                if let id = await registry.activeModelID(for: cat) {
                    activeModels[cat] = id
                }
            }
        }
    }

    private var memorySummary: String {
        let gibibytes = Double(ProcessInfo.processInfo.physicalMemory)
            / 1_073_741_824
        return "\(Int(gibibytes.rounded())) GB unified memory"
    }

    private func hardwareTierTitle(
        _ tier: DatasetAnnotationHardwareTier
    ) -> String {
        switch tier {
        case .unsupported:
            return "Native ACE not recommended"
        case .provisional:
            return "Native ACE provisional"
        case .recommended:
            return "Native ACE supported"
        }
    }

    private func hardwareTierSystemImage(
        _ tier: DatasetAnnotationHardwareTier
    ) -> String {
        switch tier {
        case .unsupported:
            return "exclamationmark.triangle"
        case .provisional:
            return "memorychip"
        case .recommended:
            return "checkmark.circle"
        }
    }

    private func revealNativeAnnotationModelFolder() {
        do {
            try FileManager.default.createDirectory(
                at: AppStorageLocations.modelsDirectory,
                withIntermediateDirectories: true
            )
            NSWorkspace.shared.open(AppStorageLocations.modelsDirectory)
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    private func downloadNativeAnnotationPack(
        task: DatasetAnnotationTask
    ) {
        guard nativePackInstallTasks[task] == nil,
              !isNativePackOperationActive
        else {
            return
        }

        nativePackInstallProgress[task] = NativeModelPackInstallProgress(
            stage: .preparing,
            fractionCompleted: 0
        )
        nativePackCancellationRequests.remove(task)
        let operation = Task { @MainActor in
            let installer = NativeModelPackInstaller(
                modelsDirectory: AppStorageLocations.modelsDirectory
            )
            do {
                _ = try await installer.install(task: task) { progress in
                    Task { @MainActor in
                        updateNativePackProgress(
                            progress,
                            task: task
                        )
                    }
                }
                refreshModels()
            } catch {
                if !isNativePackCancellation(error) {
                    errorMessage = error.localizedDescription
                }
            }
            nativePackInstallProgress[task] = nil
            nativePackCancellationRequests.remove(task)
            nativePackInstallTasks[task] = nil
        }
        nativePackInstallTasks[task] = operation
    }

    private func deleteNativeAnnotationPack(
        task: DatasetAnnotationTask
    ) {
        showNativePackDeleteConfirmation = nil
        Task { @MainActor in
            do {
                let installer = NativeModelPackInstaller(
                    modelsDirectory: AppStorageLocations.modelsDirectory
                )
                try await installer.remove(task: task)
            } catch {
                errorMessage = error.localizedDescription
            }
            refreshModels()
        }
    }

    private func updateNativePackProgress(
        _ progress: NativeModelPackInstallProgress,
        task: DatasetAnnotationTask
    ) {
        guard nativePackInstallTasks[task] != nil else { return }
        if let current = nativePackInstallProgress[task] {
            let currentRank = nativePackStageRank(current.stage)
            let newRank = nativePackStageRank(progress.stage)
            guard newRank > currentRank
                    || (
                        newRank == currentRank
                            && progress.fractionCompleted
                                >= current.fractionCompleted
                    )
            else {
                return
            }
        }
        nativePackInstallProgress[task] = progress
    }

    private func nativePackStageRank(
        _ stage: NativeModelPackInstallStage
    ) -> Int {
        switch stage {
        case .preparing: return 0
        case .downloading: return 1
        case .verifying: return 2
        case .installing: return 3
        case .complete: return 4
        }
    }

    private func cancelNativeAnnotationPackDownload(
        task: DatasetAnnotationTask
    ) {
        nativePackCancellationRequests.insert(task)
        nativePackInstallTasks[task]?.cancel()
    }

    private func isNativePackCancellation(_ error: Error) -> Bool {
        if error is CancellationError {
            return true
        }
        if let urlError = error as? URLError,
           urlError.code == .cancelled {
            return true
        }
        return Task.isCancelled
    }

    private func setActive(_ spec: ModelSpec) {
        Task {
            do {
                try await appState.registry.setActiveModel(id: spec.id, for: spec.category)
                postModelRegistryChanged()
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
                        // Each progress event spawns its own Task, and Task
                        // ordering on a MainActor isn't strictly FIFO — a late
                        // progress=1.0 event can land after refreshModels() has
                        // already pulled .ready from the registry, overwriting
                        // the terminal state with `downloading(1.0)`. Guard
                        // against downgrading a terminal state from a stale
                        // progress event.
                        switch statuses[spec.id] {
                        case .ready, .failed:
                            return
                        case .downloading(let current) where progress < current:
                            return
                        default:
                            statuses[spec.id] = .downloading(progress: progress)
                        }
                    }
                }
                postModelRegistryChanged()
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
                postModelRegistryChanged()
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
                postModelRegistryChanged()
            } catch {
                errorMessage = error.localizedDescription
            }
            refreshModels()
        }
    }

    private func postModelRegistryChanged() {
        NotificationCenter.default.post(name: .modelRegistryDidChange, object: nil)
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
                    if let label = qualityLabel {
                        Text(label)
                            .font(.caption2)
                            .fontWeight(.semibold)
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(qualityColor.opacity(0.14), in: Capsule())
                            .foregroundStyle(qualityColor)
                    }
                    if isActive {
                        Image(systemName: "checkmark.circle.fill")
                            .foregroundStyle(Color.accentColor)
                            .font(.caption)
                    }
                }
                Text(spec.totalSizeFormatted)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                if let description = tierDescription {
                    Text(description)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
                if !spec.isRunnable, let note = spec.runtimeNote {
                    Text(note)
                        .font(.caption2)
                        .foregroundStyle(.orange)
                }
            }

            Spacer()

            statusBadge

            actions
        }
        .padding(.vertical, 2)
        .listRowBackground(isActive ? Color.accentColor.opacity(0.08) : nil)
    }

    private var qualityLabel: String? {
        if !spec.isRunnable { return "Download only" }
        switch spec.id {
        case "ace-step-dit-turbo": return "Good / fastest"
        case "ace-step-dit-base":  return "Better"
        case "ace-step-dit-sft":   return "Best"
        case "ace-step-dit-xl-turbo": return "XL / fastest"
        case "ace-step-dit-xl-base":  return "XL / base"
        case "ace-step-dit-xl-sft":   return "XL / best"
        case "ace-step-vae":       return "Shared"
        case let id where id.contains("xl"): return "XL / experimental"
        default:                   return nil
        }
    }

    private var qualityColor: Color {
        if !spec.isRunnable { return .orange }
        switch spec.id {
        case "ace-step-dit-turbo": return .blue
        case "ace-step-dit-base":  return .orange
        case "ace-step-dit-sft":   return .purple
        case "ace-step-dit-xl-turbo": return .blue
        case "ace-step-dit-xl-base":  return .orange
        case "ace-step-dit-xl-sft":   return .purple
        case "ace-step-vae":       return .secondary
        case let id where id.contains("xl"): return .purple
        default:                   return .secondary
        }
    }

    private var tierDescription: String? {
        if !spec.isRunnable {
            return "Checkpoint layout is supported for download, but runtime architecture support is pending."
        }
        switch spec.id {
        case "ace-step-dit-turbo":
            return "Fast 8-step music model; best for quick iteration."
        case "ace-step-dit-base":
            return "Full 50-step base model; slower, usually more coherent than Turbo."
        case "ace-step-dit-sft":
            return "Instruction-tuned 50-step model; intended highest-quality text-to-music tier."
        case "ace-step-dit-xl-turbo":
            return "4B distilled XL model for faster high-capacity generation."
        case "ace-step-dit-xl-base":
            return "4B XL base model; broadest fine-tuning starting point."
        case "ace-step-dit-xl-sft":
            return "4B XL instruction-tuned model; quality-focused text-to-music tier."
        case "ace-step-vae":
            return "Shared VAE decoder + Qwen text encoder used by Turbo, Full, and SFT."
        case let id where id.contains("xl"):
            return "4B DiT checkpoint; higher-capacity but much slower and heavier than SFT."
        default:
            return nil
        }
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
                if !isActive && spec.isRunnable {
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
