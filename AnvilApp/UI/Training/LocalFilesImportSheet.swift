import SwiftUI
import UniformTypeIdentifiers

struct LocalFilesImportSheet: View {
    @EnvironmentObject private var outputManager: OutputDirectoryManager
    @Environment(\.dismiss) private var dismiss

    var onOpenInTraining: ([TrainingImportedFile]) -> Void

    @StateObject private var builder = LocalFilesDatasetBuilder()

    @State private var selectedFiles: [URL] = []
    @State private var selectedFolders: [URL] = []
    @State private var styleTemplate = ""
    @State private var config = LocalFilesDatasetConfig()
    @State private var editableClips: [TrainingImportedFile] = []
    @State private var showingFileImporter = false
    @State private var sourceNotice: String?
    @State private var packageSummary: String?
    @State private var isExportingPackage = false

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    filesSection
                    styleSection
                    configSection
                    switch builder.phase {
                    case .idle:
                        EmptyView()
                    case .failed(let msg):
                        errorView(msg)
                    case .chunking:
                        progressSection
                    case .complete:
                        resultsSection
                    }
                }
                .padding(16)
            }
            Divider()
            actionBar
        }
        .frame(minWidth: 540, minHeight: 500)
        .fileImporter(
            isPresented: $showingFileImporter,
            allowedContentTypes: [.audio],
            allowsMultipleSelection: true
        ) { result in
            if case .success(let urls) = result {
                appendAudioFiles(urls)
            } else if case .failure(let error) = result {
                sourceNotice = "Could not add audio files: \(error.localizedDescription)"
            }
        }
        .onChange(of: builder.phase) { _, newPhase in
            if case .complete(let dir, _) = newPhase {
                editableClips = builder.resultClips
                refreshPackageSummary(for: dir)
            }
        }
    }

    private var header: some View {
        HStack(spacing: 10) {
            Image(systemName: "folder.fill.badge.plus")
                .font(.title2)
                .foregroundStyle(.blue)
            VStack(alignment: .leading, spacing: 2) {
                Text("Build Dataset from Local Files")
                    .font(.headline)
                Text("Slice local audio into a packaged training dataset.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Button("Close") { dismiss() }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .disabled(builder.phase.isRunning)
        }
        .padding(16)
    }

    private var filesSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Source Files")
                    .font(.subheadline)
                    .fontWeight(.semibold)
                Spacer()
                Button("Add Folder...") { openFolderPanel() }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                    .disabled(builder.phase.isRunning)
                Button("Add Audio Files...") { showingFileImporter = true }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                    .disabled(builder.phase.isRunning)
            }

            if selectedFiles.isEmpty {
                Text("No files selected.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .padding(10)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(.quaternary.opacity(0.3))
                    .clipShape(RoundedRectangle(cornerRadius: 8))
            } else {
                VStack(spacing: 4) {
                    ForEach(selectedFiles.indices, id: \.self) { idx in
                        HStack(spacing: 8) {
                            Image(systemName: "waveform")
                                .foregroundStyle(.secondary)
                                .frame(width: 16)
                            Text(selectedFiles[idx].lastPathComponent)
                                .font(.callout)
                                .lineLimit(1)
                                .truncationMode(.middle)
                            Spacer()
                            Button {
                                selectedFiles.remove(at: idx)
                            } label: {
                                Image(systemName: "xmark.circle")
                                    .foregroundStyle(.secondary)
                            }
                            .buttonStyle(.borderless)
                            .disabled(builder.phase.isRunning)
                        }
                    }
                }
                .padding(10)
                    .background(.quaternary.opacity(0.3))
                    .clipShape(RoundedRectangle(cornerRadius: 8))
            }

            if let sourceNotice {
                Text(sourceNotice)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }

    private var styleSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Style Description")
                .font(.subheadline)
                .fontWeight(.semibold)
            TextField("dark blues, slow tempo, minor key, atmospheric guitar", text: $styleTemplate)
                .textFieldStyle(.roundedBorder)
                .disabled(builder.phase.isRunning)
            Text("Applied to every clip as its training caption. Leave blank to use file names.")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    private var configSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Import Settings")
                .font(.subheadline)
                .fontWeight(.semibold)

            VStack(spacing: 6) {
                LabeledContent("Clip Length") {
                    HStack(spacing: 8) {
                        Slider(value: $config.clipLength, in: 15...60, step: 5)
                            .frame(maxWidth: 160)
                            .disabled(builder.phase.isRunning)
                        Text("\(Int(config.clipLength))s")
                            .monospacedDigit()
                            .frame(width: 34, alignment: .trailing)
                            .foregroundStyle(.secondary)
                    }
                }
                LabeledContent("Clips per File") {
                    Stepper("\(config.clipsPerTrack)",
                            value: $config.clipsPerTrack,
                            in: 1...20)
                        .disabled(builder.phase.isRunning)
                        .frame(maxWidth: 130)
                }
                LabeledContent("Max Total Clips") {
                    Stepper("\(config.maxTotalClips)",
                            value: $config.maxTotalClips,
                            in: 5...200,
                            step: 5)
                        .disabled(builder.phase.isRunning)
                        .frame(maxWidth: 130)
                }
            }
            .padding(10)
            .background(.quaternary.opacity(0.3))
            .clipShape(RoundedRectangle(cornerRadius: 8))

            if let dir = outputManager.directory {
                HStack(spacing: 4) {
                    Image(systemName: "folder")
                        .foregroundStyle(.secondary)
                    Text("Output: \(dir.path)/TrainingDatasets/...")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
            } else {
                HStack(spacing: 4) {
                    Image(systemName: "exclamationmark.triangle")
                        .foregroundStyle(.orange)
                    Text("No output directory set. Configure one in Settings before starting.")
                        .font(.caption)
                        .foregroundStyle(.orange)
                }
            }
        }
    }

    private var progressSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Importing")
                .font(.subheadline)
                .fontWeight(.semibold)

            HStack(spacing: 12) {
                Image(systemName: "scissors")
                    .font(.title2)
                    .foregroundStyle(.orange)
                VStack(alignment: .leading, spacing: 4) {
                    Text(progressLabel)
                        .font(.callout)
                        .fontWeight(.medium)
                    Text(progressDetail)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                    progressIndicator
                }
            }
            .padding(12)
            .background(.quaternary.opacity(0.3))
            .clipShape(RoundedRectangle(cornerRadius: 8))
        }
    }

    private var progressLabel: String {
        switch builder.phase {
        case .chunking(let idx, let total, _):
            return idx == 0 ? "Preparing files..." : "Chunking \(idx) of \(total)"
        default:
            return ""
        }
    }

    private var progressDetail: String {
        switch builder.phase {
        case .chunking(_, _, let name): return name
        default: return ""
        }
    }

    @ViewBuilder
    private var progressIndicator: some View {
        switch builder.phase {
        case .chunking(let idx, let total, _):
            ProgressView(value: Double(max(0, idx - 1)), total: Double(max(1, total)))
                .frame(maxWidth: .infinity)
        default:
            EmptyView()
        }
    }

    private var resultsSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 8) {
                Image(systemName: "checkmark.circle.fill")
                    .foregroundStyle(.green)
                Text("\(editableClips.count) clips packaged for training")
                    .font(.callout)
                    .fontWeight(.medium)
                Spacer()
                if case .complete(let dir, _) = builder.phase {
                    Button {
                        NSWorkspace.shared.open(dir)
                    } label: {
                        Label("Show in Finder", systemImage: "folder")
                            .font(.caption)
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                    Button {
                        exportPackage(from: dir)
                    } label: {
                        Label("Export Package...", systemImage: "archivebox")
                            .font(.caption)
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                    .disabled(isExportingPackage)
                }
            }
            .padding(10)
            .background(.green.opacity(0.08))
            .clipShape(RoundedRectangle(cornerRadius: 8))

            if let packageSummary {
                Text(packageSummary)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Text("Review and edit captions below. The package manifest and training bundles are refreshed when opened.")
                .font(.caption)
                .foregroundStyle(.secondary)

            VStack(spacing: 4) {
                ForEach(editableClips.indices, id: \.self) { idx in
                    HStack(spacing: 8) {
                        Image(systemName: "waveform")
                            .foregroundStyle(.secondary)
                            .frame(width: 16)
                        TextField("Caption", text: Binding(
                            get: { editableClips[idx].description },
                            set: { editableClips[idx].description = $0 }
                        ))
                        .textFieldStyle(.roundedBorder)
                        .font(.callout)
                    }
                    .padding(.vertical, 2)
                }
            }
            .padding(10)
            .background(.quaternary.opacity(0.3))
            .clipShape(RoundedRectangle(cornerRadius: 8))
        }
    }

    private func errorView(_ message: String) -> some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: "xmark.circle.fill")
                .foregroundStyle(.red)
                .padding(.top, 1)
            VStack(alignment: .leading, spacing: 4) {
                Text("Import failed")
                    .font(.callout)
                    .fontWeight(.semibold)
                Text(message)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
        }
        .padding(12)
        .background(.red.opacity(0.08))
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }

    private var actionBar: some View {
        HStack {
            Button("Cancel") { dismiss() }
                .buttonStyle(.bordered)
                .disabled(builder.phase.isRunning)

            Spacer()

            switch builder.phase {
            case .idle, .failed:
                Button("Start") { startImport() }
                    .buttonStyle(.borderedProminent)
                    .disabled(!canStart)
            case .chunking:
                Button("Stop", role: .destructive) { builder.cancel() }
                    .buttonStyle(.bordered)
            case .complete:
                Button("Open in Training") {
                    persistCaptions()
                    onOpenInTraining(editableClips)
                }
                .buttonStyle(.borderedProminent)
            }
        }
        .padding(16)
    }

    private var canStart: Bool {
        !selectedFiles.isEmpty && outputManager.directory != nil
    }

    private func startImport() {
        guard let outputBase = outputManager.directory else { return }
        let template = styleTemplate.trimmingCharacters(in: .whitespacesAndNewlines)
        builder.start(
            sourceFiles: selectedFiles,
            sourceDirectories: selectedFolders,
            config: config,
            outputBase: outputBase,
            styleTemplate: template
        )
    }

    private func persistCaptions() {
        guard case .complete(let dir, _) = builder.phase else { return }
        try? TrainingDatasetPackageWriter.writeLocalPackage(
            outputDirectory: dir,
            clips: editableClips,
            sourceFiles: selectedFiles,
            config: config,
            styleTemplate: styleTemplate
        )
        refreshPackageSummary(for: dir)
    }

    private func refreshPackageSummary(for dir: URL) {
        packageSummary = TrainingDatasetPackageWriter.validatePackage(at: dir).summary
    }

    private func exportPackage(from dir: URL) {
        persistCaptions()

        let panel = NSSavePanel()
        panel.allowedContentTypes = [.zip]
        panel.canCreateDirectories = true
        panel.nameFieldStringValue = "\(dir.lastPathComponent).zip"
        guard panel.runModal() == .OK, let destination = panel.url else { return }

        isExportingPackage = true
        defer { isExportingPackage = false }

        do {
            let archiveURL = try TrainingDatasetPackageWriter.exportPackageArchive(
                outputDirectory: dir,
                to: destination
            )
            sourceNotice = "Package exported to \(archiveURL.lastPathComponent)."
        } catch {
            sourceNotice = error.localizedDescription
        }
    }

    private func openFolderPanel() {
        let panel = NSOpenPanel()
        panel.title = "Add Audio Folder"
        panel.prompt = "Open"
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = true
        panel.canCreateDirectories = false

        guard panel.runModal() == .OK else { return }
        appendFolders(panel.urls)
    }

    private func appendAudioFiles(_ urls: [URL]) {
        let audioURLs = urls.filter { LocalAudioFileDiscovery.supportedExtensions.contains($0.pathExtension.lowercased()) }
        let originalCount = selectedFiles.count
        selectedFiles = mergedURLs(selectedFiles, with: audioURLs)
        let addedCount = selectedFiles.count - originalCount
        sourceNotice = addedCount == 0
            ? "Selected audio files were already in the list."
            : "\(addedCount) audio file\(addedCount == 1 ? "" : "s") added."
    }

    private func appendFolders(_ urls: [URL]) {
        selectedFolders = mergedURLs(selectedFolders, with: urls)
        let discovered = LocalAudioFileDiscovery.audioFiles(in: urls)
        let originalCount = selectedFiles.count
        selectedFiles = mergedURLs(selectedFiles, with: discovered)
        let addedCount = selectedFiles.count - originalCount
        sourceNotice = addedCount == 0
            ? "No new audio files were found in the selected folder\(urls.count == 1 ? "" : "s")."
            : "\(addedCount) audio file\(addedCount == 1 ? "" : "s") added from folder\(urls.count == 1 ? "" : "s")."
    }

    private func mergedURLs(_ existing: [URL], with incoming: [URL]) -> [URL] {
        var seen = Set(existing.map { $0.standardizedFileURL.path })
        var merged = existing
        for url in incoming where seen.insert(url.standardizedFileURL.path).inserted {
            merged.append(url)
        }
        return merged
    }
}
