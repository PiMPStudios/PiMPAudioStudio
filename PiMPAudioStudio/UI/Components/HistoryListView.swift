import SwiftUI

struct HistoryListView: View {
    @EnvironmentObject private var historyStore: HistoryStore
    @EnvironmentObject private var playerService: AudioPlayerService
    @EnvironmentObject private var trainingService: TrainingService
    @EnvironmentObject private var traceMonitor: GenerationTraceMonitor
    @EnvironmentObject private var presetStore: PresetStore
    @Environment(\.openWindow) private var openWindow

    @State private var isSelecting = false
    @State private var selectedIDs: Set<UUID> = []
    @State private var showTrainingSheet = false
    @State private var traceErrorMessage: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            header
            Divider()
            if historyStore.entries.isEmpty {
                emptyState
            } else {
                list
            }
        }
        .sheet(isPresented: $showTrainingSheet) {
            let selected = historyStore.entries.filter { selectedIDs.contains($0.id) }
            TrainingSheet(selectedEntries: selected)
        }
    }

    // MARK: - Header

    private var header: some View {
        HStack {
            Text("HISTORY")
                .font(.caption)
                .fontWeight(.semibold)
                .foregroundStyle(.secondary)

            if isSelecting && !selectedIDs.isEmpty {
                Text("\(selectedIDs.count) selected")
                    .font(.caption2)
                    .foregroundStyle(.blue)
            }

            Spacer()

            if !historyStore.entries.isEmpty {
                if isSelecting {
                    if !selectedIDs.isEmpty {
                        if selectedIDs.count == 1 {
                            Button("Open in Generate") {
                                openSelectedInGenerate()
                            }
                            .buttonStyle(.borderedProminent)
                            .controlSize(.mini)
                            .help("Reload this take’s prompt, seed, guidance, and adapters into Generate for a quick tweak")

                            Button("Edit") {
                                openSelectedInEditor()
                            }
                            .buttonStyle(.bordered)
                            .controlSize(.mini)
                            .help("Load this take into the Editor for polish and DSP")
                            .disabled(!canOpenSelectedInEditor)
                        }

                        if selectedIDs.count == 2 {
                            Button("Compare") {
                                compareSelected()
                            }
                            .buttonStyle(.borderedProminent)
                            .controlSize(.mini)
                            .disabled(!canCompareSelected)
                        }

                        Button("Train LoRA") {
                            showTrainingSheet = true
                        }
                        .buttonStyle(.borderedProminent)
                        .controlSize(.mini)

                        // Export all selected items to a chosen folder.
                        Button("Export…") {
                            exportSelected()
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.mini)

                        Button(role: .destructive) {
                            deleteSelected()
                        } label: {
                            Label("Delete", systemImage: "trash")
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.mini)
                    }

                    Button("Done") {
                        isSelecting = false
                        selectedIDs.removeAll()
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.mini)
                } else {
                    Button("Select") {
                        isSelecting = true
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.mini)

                    Button("Delete All", role: .destructive) {
                        historyStore.deleteAll()
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.mini)
                }
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
    }

    // MARK: - Empty state

    private var emptyState: some View {
        VStack(spacing: 8) {
            Image(systemName: "waveform")
                .font(.title)
                .foregroundStyle(.quaternary)
            Text("No generations yet")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    // MARK: - List

    private var list: some View {
        List {
            ForEach(historyStore.entries) { entry in
                HStack(spacing: 8) {
                    if isSelecting {
                        Image(systemName: selectedIDs.contains(entry.id) ? "checkmark.circle.fill" : "circle")
                            .foregroundStyle(selectedIDs.contains(entry.id) ? Color.accentColor : .secondary)
                    }
                    HistoryRow(entry: entry)
                    if !isSelecting && canAnalyze(entry) {
                        Spacer(minLength: 8)
                        Button {
                            analyzeTrace(for: entry)
                        } label: {
                            Image(systemName: "waveform.path.ecg")
                        }
                        .buttonStyle(.borderless)
                        .help("Open trace in Generation Analyzer")
                    }
                }
                .contentShape(Rectangle())
                .onTapGesture {
                    if isSelecting {
                        toggleSelection(entry.id)
                    } else if let url = try? historyStore.fileURL(for: entry) {
                        playerService.load(url: url)
                    }
                }
                .contextMenu {
                    rowContextMenu(for: entry)
                }
            }
            .onDelete { indices in
                let toDelete = indices.map { historyStore.entries[$0] }
                for entry in toDelete {
                    historyStore.delete(entry)
                }
            }
        }
        .listStyle(.plain)
        .alert("Could not open trace", isPresented: Binding(
            get: { traceErrorMessage != nil },
            set: { if !$0 { traceErrorMessage = nil } }
        )) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(traceErrorMessage ?? "")
        }
    }

    // MARK: - Context menu

    @ViewBuilder
    private func rowContextMenu(for entry: HistoryEntry) -> some View {
        if let url = try? historyStore.fileURL(for: entry) {
            Button("Play") {
                playerService.load(url: url)
            }
        }

        Button("Open in Generate") {
            openInGenerate(entry)
        }

        if (try? historyStore.fileURL(for: entry)) != nil {
            Button("Edit") {
                openInEditor(entry)
            }
        }

        if canAnalyze(entry) {
            Button("Analyze Trace") {
                analyzeTrace(for: entry)
            }

            Button("Reveal Trace in Finder") {
                if let traceURL = historyStore.traceURL(for: entry) {
                    NSWorkspace.shared.activateFileViewerSelecting([traceURL])
                }
            }
        }

        Divider()

        if let url = try? historyStore.fileURL(for: entry) {
            ShareLink(
                item: url,
                preview: SharePreview(
                    entry.prompt,
                    icon: Image(systemName: "waveform")
                )
            ) {
                Label("Share…", systemImage: "square.and.arrow.up")
            }

            Button("Export as WAV…") {
                exportEntry(entry)
            }

            Button("Reveal in Finder") {
                NSWorkspace.shared.activateFileViewerSelecting([url])
            }

            Divider()
        }

        Button("Delete", role: .destructive) {
            historyStore.delete(entry)
        }
    }

    // MARK: - Helpers

    private func toggleSelection(_ id: UUID) {
        if selectedIDs.contains(id) {
            selectedIDs.remove(id)
        } else {
            selectedIDs.insert(id)
        }
    }

    private func canAnalyze(_ entry: HistoryEntry) -> Bool {
        historyStore.traceURL(for: entry) != nil
    }

    private func analyzeTrace(for entry: HistoryEntry) {
        do {
            let trace = try historyStore.loadTrace(for: entry)
            traceMonitor.inspect(trace: trace)
            openWindow(id: "generationAnalyzer")
        } catch {
            traceErrorMessage = error.localizedDescription
        }
    }

    private var selectedEntries: [HistoryEntry] {
        historyStore.entries.filter { selectedIDs.contains($0.id) }
    }

    private var canCompareSelected: Bool {
        let entries = selectedEntries
        guard entries.count == 2 else { return false }
        return entries.allSatisfy { historyStore.traceURL(for: $0) != nil }
    }

    private var canOpenSelectedInEditor: Bool {
        guard let entry = selectedEntries.first, selectedEntries.count == 1 else { return false }
        return (try? historyStore.fileURL(for: entry)) != nil
    }

    private func openSelectedInGenerate() {
        guard selectedIDs.count == 1, let entry = selectedEntries.first else { return }
        openInGenerate(entry)
        isSelecting = false
        selectedIDs.removeAll()
    }

    private func openSelectedInEditor() {
        guard selectedIDs.count == 1, let entry = selectedEntries.first else { return }
        openInEditor(entry)
        isSelecting = false
        selectedIDs.removeAll()
    }

    private func openInGenerate(_ entry: HistoryEntry) {
        let trace = try? historyStore.loadTrace(for: entry)
        let preset = entry.asGenerationPreset(
            name: "From History",
            trace: trace
        )
        let stack = entry.adapterStackPayload(trace: trace)
        // Navigate to Generate and restore knobs + adapter stack (see GenerateView).
        presetStore.historyRestore = HistoryGenerateRestore(
            preset: preset,
            adapterStack: stack
        )
    }

    private func openInEditor(_ entry: HistoryEntry) {
        guard let url = try? historyStore.fileURL(for: entry) else {
            traceErrorMessage = HistoryStoreError.invalidFileName(entry.fileName).localizedDescription
            return
        }
        playerService.pause()
        NotificationCenter.default.post(
            name: .navigateToEditor,
            object: nil,
            userInfo: [
                "url": url,
                "prompt": entry.originalPrompt ?? entry.prompt,
            ]
        )
    }

    private func compareSelected() {
        let entries = selectedEntries
        guard entries.count == 2 else { return }

        do {
            let traces = try entries.map { try historyStore.loadTrace(for: $0) }
            traceMonitor.compare(left: traces[0], right: traces[1])
            openWindow(id: "generationAnalyzer")
        } catch {
            traceErrorMessage = error.localizedDescription
        }
    }

    private func deleteSelected() {
        let ids = selectedIDs
        guard !ids.isEmpty else { return }

        historyStore.delete(ids: ids)
        selectedIDs.removeAll()
        if historyStore.entries.isEmpty {
            isSelecting = false
        }
    }

    /// Save a single history entry to a user-chosen path via NSSavePanel.
    private func exportEntry(_ entry: HistoryEntry) {
        guard let source = try? historyStore.fileURL(for: entry) else {
            traceErrorMessage = HistoryStoreError.invalidFileName(entry.fileName).localizedDescription
            return
        }
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.wav]
        panel.nameFieldStringValue = entry.fileName
        panel.begin { response in
            guard response == .OK, let destination = panel.url else { return }
            try? FileManager.default.copyItem(at: source, to: destination)
        }
    }

    /// Copy all selected entries into a user-chosen folder via NSOpenPanel.
    private func exportSelected() {
        let sources = historyStore.entries
            .filter { selectedIDs.contains($0.id) }
            .compactMap { try? historyStore.fileURL(for: $0) }
        guard !sources.isEmpty else { return }

        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.canCreateDirectories = true
        panel.allowsMultipleSelection = false
        panel.prompt = "Export Here"
        panel.message = "Choose a folder to copy \(sources.count) file(s) into."

        panel.begin { response in
            guard response == .OK, let destination = panel.url else { return }
            for source in sources {
                let dest = destination.appending(path: source.lastPathComponent)
                // Avoid overwriting: append a numeric suffix if needed.
                let safeURL = uniqueURL(dest)
                try? FileManager.default.copyItem(at: source, to: safeURL)
            }
        }
    }

    /// Returns a URL that doesn't clash with an existing file by appending -2, -3, …
    private func uniqueURL(_ url: URL) -> URL {
        guard FileManager.default.fileExists(atPath: url.path(percentEncoded: false)) else {
            return url
        }
        let base = url.deletingPathExtension().lastPathComponent
        let ext  = url.pathExtension
        let dir  = url.deletingLastPathComponent()
        var n = 2
        while true {
            let candidate = dir.appending(path: "\(base)-\(n).\(ext)")
            if !FileManager.default.fileExists(atPath: candidate.path(percentEncoded: false)) {
                return candidate
            }
            n += 1
        }
    }
}

// MARK: - HistoryRow

struct HistoryRow: View {
    let entry: HistoryEntry

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(entry.prompt)
                .font(.callout)
                .lineLimit(1)
                .truncationMode(.tail)

            HStack(spacing: 8) {
                Text("\(Int(entry.duration))s")
                Text("\u{2022}")
                Text(entry.createdAt, style: .date)
                Text("\u{2022}")
                Text("seed: \(entry.seed)")
                if entry.traceFileName != nil {
                    Label("Trace", systemImage: "waveform.path.ecg")
                        .labelStyle(.iconOnly)
                        .foregroundStyle(.blue)
                        .help("Trace sidecar available")
                }
                if entry.isPreferred == true {
                    Label("Preferred", systemImage: "star.fill")
                        .labelStyle(.iconOnly)
                        .foregroundStyle(.yellow)
                        .help("Preferred take")
                }
                if !(entry.listeningNote ?? "").trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    Label("Note", systemImage: "note.text")
                        .labelStyle(.iconOnly)
                        .foregroundStyle(.secondary)
                        .help(entry.listeningNote ?? "")
                }
            }
            .font(.caption2)
            .foregroundStyle(.secondary)
        }
        .padding(.vertical, 4)
    }
}
