import SwiftUI
import AnvilCore

struct AdaptersSettingsView: View {
    enum Tab: String, CaseIterable {
        case myAdapters = "My Adapters"
        case community = "Community"
    }

    @State private var selectedTab: Tab = .myAdapters

    var body: some View {
        VStack(spacing: 0) {
            // Segmented control
            Picker("", selection: $selectedTab) {
                ForEach(Tab.allCases, id: \.self) { tab in
                    Text(tab.rawValue).tag(tab)
                }
            }
            .pickerStyle(.segmented)
            .padding(.horizontal)
            .padding(.top, 12)
            .padding(.bottom, 8)

            Divider()

            // Tab content
            switch selectedTab {
            case .myAdapters:
                MyAdaptersListView()
            case .community:
                CommunityBrowserView()
            }
        }
    }
}

// MARK: - My Adapters (extracted from original)

/// Drives `.sheet(item:)` for any training sheet presentation.
/// Carrying all data as value types ensures the sheet closure sees the
/// correct dataset regardless of when SwiftUI evaluates it.
private struct TrainingPresentation: Identifiable {
    let id = UUID()
    /// History entries resolved from the manifest (may be empty for new sessions).
    let selectedEntries: [HistoryEntry]
    /// Imported file entries — from manifest, YouTube, or manual selection.
    let files: [TrainingImportedFile]
    /// Pre-fill the adapter name field (empty = blank for new sessions).
    let initialName: String
    /// Pre-fill training config (nil = use defaults for new sessions).
    let initialConfig: LoRAConfig?
    /// Adapter ID to resume from a saved checkpoint. Nil means fresh training/retraining.
    let resumeAdapterID: String?

    /// Blank new training session.
    init(files: [TrainingImportedFile] = []) {
        self.selectedEntries = []
        self.files = files
        self.initialName = ""
        self.initialConfig = nil
        self.resumeAdapterID = nil
    }

    /// Pre-populated retrain session.
    init(
        selectedEntries: [HistoryEntry],
        files: [TrainingImportedFile],
        initialName: String,
        initialConfig: LoRAConfig?,
        resumeAdapterID: String? = nil
    ) {
        self.selectedEntries = selectedEntries
        self.files = files
        self.initialName = initialName
        self.initialConfig = initialConfig
        self.resumeAdapterID = resumeAdapterID
    }
}

private struct MyAdaptersListView: View {
    @EnvironmentObject private var outputManager: OutputDirectoryManager
    @EnvironmentObject private var historyStore: HistoryStore

    @State private var adapters: [LoRAAdapter] = []
    @State private var showingMergeSheet = false
    @State private var trainingPresentation: TrainingPresentation?
    @State private var showingYouTubeSheet = false
    @State private var showingLocalFilesSheet = false
    @State private var renamingAdapter: LoRAAdapter?
    @State private var renameText: String = ""
    @State private var errorMessage: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Toolbar
            HStack {
                Text("LoRA Adapters")
                    .font(.headline)
                Spacer()
                Button("From YouTube…") { showingYouTubeSheet = true }
                    .buttonStyle(.bordered)
                Button("From Local Files…") { showingLocalFilesSheet = true }
                    .buttonStyle(.bordered)
                Button("Train from Files…") { trainingPresentation = TrainingPresentation() }
                    .buttonStyle(.bordered)
                Button("Merge Two") {
                    showingMergeSheet = true
                }
                .disabled(adapters.count < 2)
            }
            .padding()

            Divider()

            if adapters.isEmpty {
                emptyState
            } else {
                adapterList
            }

            // Error banner
            if let error = errorMessage {
                HStack {
                    Image(systemName: "exclamationmark.triangle")
                    Text(error)
                    Spacer()
                    Button("Dismiss") { errorMessage = nil }
                        .buttonStyle(.borderless)
                }
                .font(.caption)
                .foregroundStyle(.red)
                .padding(8)
                .background(.red.opacity(0.05))
            }
        }
        .onAppear { refreshAdapters() }
        .onReceive(NotificationCenter.default.publisher(for: .adapterRegistryDidChange)) { _ in
            refreshAdapters()
        }
        .sheet(isPresented: $showingMergeSheet) {
            MergeSheet(onComplete: { refreshAdapters() })
                .frame(minWidth: 400, minHeight: 300)
        }
        .sheet(item: $trainingPresentation, onDismiss: { refreshAdapters() }) { presentation in
            TrainingSheet(
                selectedEntries: presentation.selectedEntries,
                initialImportedFiles: presentation.files,
                initialName: presentation.initialName,
                initialConfig: presentation.initialConfig,
                resumeAdapterID: presentation.resumeAdapterID
            )
        }
        .sheet(isPresented: $showingYouTubeSheet) {
            YouTubeImportSheet { files in
                showingYouTubeSheet = false
                DispatchQueue.main.async {
                    trainingPresentation = TrainingPresentation(files: files)
                }
            }
            .environmentObject(outputManager)
        }
        .sheet(isPresented: $showingLocalFilesSheet) {
            LocalFilesImportSheet { files in
                showingLocalFilesSheet = false
                DispatchQueue.main.async {
                    trainingPresentation = TrainingPresentation(files: files)
                }
            }
            .environmentObject(outputManager)
        }
        .alert("Rename Adapter", isPresented: Binding(
            get: { renamingAdapter != nil },
            set: { if !$0 { renamingAdapter = nil } }
        )) {
            TextField("New name", text: $renameText)
            Button("Cancel", role: .cancel) { renamingAdapter = nil }
            Button("Rename") { performRename() }
        } message: {
            Text("Enter a new name for \"\(renamingAdapter?.name ?? "")\"")
        }
    }

    private var emptyState: some View {
        VStack(spacing: 8) {
            Spacer()
            Image(systemName: "puzzlepiece.extension")
                .font(.largeTitle)
                .foregroundStyle(.secondary)
            Text("No Adapters Yet")
                .font(.headline)
                .foregroundStyle(.secondary)
            Text("Train one from History or download from the Community tab.")
                .font(.caption)
                .foregroundStyle(.tertiary)
            Spacer()
        }
        .frame(maxWidth: .infinity)
        .padding()
    }

    private var adapterList: some View {
        List {
            ForEach(adapters) { adapter in
                adapterRow(adapter)
                    .contextMenu {
                        if AdapterRegistry.hasCheckpoint(adapterID: adapter.id) {
                            Button("Resume Training…") { beginResume(adapter) }
                        }
                        Button("Retrain…") { beginRetrain(adapter) }
                        Divider()
                        Button("Rename...") {
                            renameText = adapter.name
                            renamingAdapter = adapter
                        }
                        Button("Export...") { beginExport(adapter) }
                        Divider()
                        Button("Delete", role: .destructive) { deleteAdapter(adapter) }
                    }
            }
            .onDelete { indexSet in
                for idx in indexSet {
                    deleteAdapter(adapters[idx])
                }
            }
        }
    }

    private func adapterRow(_ adapter: LoRAAdapter) -> some View {
        let hasCheckpoint = AdapterRegistry.hasCheckpoint(adapterID: adapter.id)

        return HStack {
            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 6) {
                    Text(adapter.name)
                        .font(.body)
                        .fontWeight(.medium)
                    if hasCheckpoint {
                        Text("In Progress")
                            .font(.caption2)
                            .fontWeight(.medium)
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(.orange.opacity(0.15))
                            .foregroundStyle(.orange)
                            .clipShape(Capsule())
                    }
                    // Source badge
                    if adapter.source != .local {
                        Text(adapter.source.rawValue.capitalized)
                            .font(.caption2)
                            .padding(.horizontal, 5)
                            .padding(.vertical, 1)
                            .background(.blue.opacity(0.1))
                            .clipShape(Capsule())
                    }
                }
                HStack(spacing: 8) {
                    Text(adapter.baseModelCategory.rawValue)
                        .font(.caption2)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(.blue.opacity(0.1))
                        .clipShape(Capsule())
                    if let variant = modelVariant(for: adapter.baseModelID) {
                        Text(variant)
                            .font(.caption2)
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(.secondary.opacity(0.15))
                            .clipShape(Capsule())
                    }
                    Text("Rank \(adapter.config.rank)")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    if let loss = adapter.finalLoss {
                        Text("Loss: \(String(format: "%.4f", loss))")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                }
            }
            Spacer()
            Text(adapter.createdAt, style: .date)
                .font(.caption)
                .foregroundStyle(.tertiary)
        }
        .padding(.vertical, 4)
    }

    // MARK: - Helpers

    /// Short display name for the model variant, or nil for unknown / non-ACE-Step IDs.
    private func modelVariant(for baseModelID: String) -> String? {
        switch baseModelID {
        case "ace-step-dit-turbo": return "Turbo"
        case "ace-step-dit-sft":   return "SFT"
        case "ace-step-dit-base":  return "Base"
        case "ace-step-dit-xl":    return "XL"
        default:                   return nil
        }
    }

    // MARK: - Actions

    private func refreshAdapters() {
        adapters = AdapterRegistry.listAdapters()
    }

    private func resolvedTrainingSources(
        for adapter: LoRAAdapter
    ) -> (entries: [HistoryEntry], files: [TrainingImportedFile]) {
        let manifest = AdapterRegistry.loadManifest(adapterID: adapter.id)

        // Resolve history entries still present in the store
        let resolvedEntries: [HistoryEntry]
        if let ids = manifest?.historyEntryIDs {
            let idSet = Set(ids)
            resolvedEntries = historyStore.entries.filter { idSet.contains($0.id) }
        } else {
            resolvedEntries = []
        }

        // Resolve imported file references still accessible on disk
        let resolvedFiles: [TrainingImportedFile]
        if let refs = manifest?.importedFiles {
            resolvedFiles = refs.compactMap { ref in
                FileManager.default.fileExists(atPath: ref.url.path(percentEncoded: false))
                    ? ref
                    : nil
            }
        } else {
            resolvedFiles = []
        }

        return (resolvedEntries, resolvedFiles)
    }

    /// Opens the training sheet pre-populated from the adapter's saved manifest.
    ///
    /// - History entries are resolved from the live `historyStore`; any that were
    ///   deleted since training are silently skipped.
    /// - Imported file references whose paths no longer exist on disk are skipped.
    /// - If no manifest was saved (adapter trained before this feature), the sheet
    ///   opens with just the name and config pre-filled; the dataset will be empty.
    private func beginRetrain(_ adapter: LoRAAdapter) {
        let sources = resolvedTrainingSources(for: adapter)

        trainingPresentation = TrainingPresentation(
            selectedEntries: sources.entries,
            files: sources.files,
            initialName: adapter.name,
            initialConfig: adapter.config
        )
    }

    private func beginResume(_ adapter: LoRAAdapter) {
        guard let checkpoint = AdapterRegistry.loadCheckpoint(adapterID: adapter.id) else {
            errorMessage = "No paused checkpoint was found for \(adapter.name)."
            refreshAdapters()
            return
        }

        let sources = resolvedTrainingSources(for: adapter)
        guard !sources.entries.isEmpty || !sources.files.isEmpty else {
            errorMessage = "Cannot resume \(adapter.name) because its original training files are missing."
            return
        }

        trainingPresentation = TrainingPresentation(
            selectedEntries: sources.entries,
            files: sources.files,
            initialName: adapter.name,
            initialConfig: checkpoint.config,
            resumeAdapterID: adapter.id
        )
    }

    private func performRename() {
        guard let adapter = renamingAdapter else { return }
        do {
            try AdapterRegistry.rename(adapterID: adapter.id, to: renameText)
            refreshAdapters()
        } catch {
            errorMessage = error.localizedDescription
        }
        renamingAdapter = nil
    }

    private func deleteAdapter(_ adapter: LoRAAdapter) {
        do {
            try AdapterRegistry.delete(adapterID: adapter.id)
            refreshAdapters()
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    private func beginExport(_ adapter: LoRAAdapter) {
        let panel = NSSavePanel()
        panel.title = "Export Adapter"
        panel.nameFieldStringValue = adapter.name
        panel.canCreateDirectories = true

        if panel.runModal() == .OK, let url = panel.url {
            do {
                try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
                try AdapterRegistry.export(adapterID: adapter.id, to: url)
            } catch {
                errorMessage = error.localizedDescription
            }
        }
    }
}
