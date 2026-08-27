import SwiftUI
import UniformTypeIdentifiers

struct LocalFilesImportSheet: View {
    @EnvironmentObject private var outputManager: OutputDirectoryManager
    @EnvironmentObject private var generationService: GenerationService
    @Environment(\.dismiss) private var dismiss

    var onOpenDatasetReview: (URL) -> Void

    @StateObject private var builder = LocalFilesDatasetBuilder()
    @StateObject private var enrichment = LocalDatasetEnrichmentService()
    @State private var enrichmentOptions = LocalDatasetEnrichmentOptions()

    @State private var selectedFiles: [URL] = []
    @State private var selectedFolders: [URL] = []
    @State private var datasetName = ""
    @State private var styleTemplate = ""
    @State private var config = LocalFilesDatasetConfig()
    @State private var editableClips: [TrainingImportedFile] = []
    @State private var showingFileImporter = false
    @State private var sourceNotice: String?
    @State private var packageSummary: String?
    @State private var isExportingPackage = false
    @State private var hasUnsavedCaptionEdits = false

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    datasetNameSection
                    filesSection
                    styleSection
                    configSection
                    switch builder.phase {
                    case .idle:
                        EmptyView()
                    case let .failed(msg):
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
            if case let .success(urls) = result {
                appendAudioFiles(urls)
            } else if case let .failure(error) = result {
                sourceNotice = "Could not add audio files: \(error.localizedDescription)"
            }
        }
        .onChange(of: builder.phase) { _, newPhase in
            if case let .complete(dir, _) = newPhase {
                reloadEditableClips(from: dir, fallback: builder.resultClips)
                hasUnsavedCaptionEdits = false
                enrichment.reset()
                refreshPackageSummary(for: dir)
            }
        }
        .onChange(of: enrichment.phase) { _, newPhase in
            if case let .complete(result) = newPhase {
                reloadEditableClips(from: result.datasetDirectory)
                hasUnsavedCaptionEdits = false
                refreshPackageSummary(for: result.datasetDirectory)
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
                Text("Package complete tracks or extract clips for training.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Button("Close") { dismiss() }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .disabled(isBusy)
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
                    .disabled(isBusy)
                Button("Add Audio Files...") { showingFileImporter = true }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                    .disabled(isBusy)
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
                            .disabled(isBusy)
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

    private var datasetNameSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Dataset Name")
                .font(.subheadline)
                .fontWeight(.semibold)
            TextField("Dark Gentlemen vocal dataset", text: $datasetName)
                .textFieldStyle(.roundedBorder)
                .disabled(isBusy)
            Text("Used for the package folder and exported archive name. Leave blank to keep the first-file fallback.")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    private var styleSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Style Description")
                .font(.subheadline)
                .fontWeight(.semibold)
            TextField("dark blues, slow tempo, minor key, atmospheric guitar", text: $styleTemplate)
                .textFieldStyle(.roundedBorder)
                .disabled(isBusy)
            Text(
                "Applied to every training item as its caption. Leave blank for filename placeholders that Enrich Dataset can replace with native metadata captions."
            )
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
                HStack(spacing: 10) {
                    Text("Source Mode")
                        .frame(width: 130, alignment: .trailing)
                    Picker("Source Mode", selection: $config.buildMode) {
                        ForEach(LocalDatasetBuildMode.allCases) { mode in
                            Text(mode.displayName).tag(mode)
                        }
                    }
                    .labelsHidden()
                    .pickerStyle(.segmented)
                    .disabled(isBusy)
                    .frame(width: 300)
                }
                Text(config.buildMode.helpText)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)

                if config.buildMode == .clipExtraction {
                    LabeledContent("Clip Length") {
                        HStack(spacing: 8) {
                            Slider(value: $config.clipLength, in: 15 ... 60, step: 5)
                                .frame(maxWidth: 160)
                                .disabled(isBusy)
                            Text("\(Int(config.clipLength))s")
                                .monospacedDigit()
                                .frame(width: 34, alignment: .trailing)
                                .foregroundStyle(.secondary)
                        }
                    }
                    LabeledContent("Clips per File") {
                        Stepper(
                            "\(config.clipsPerTrack)",
                            value: $config.clipsPerTrack,
                            in: 1 ... 20
                        )
                        .disabled(isBusy)
                        .frame(maxWidth: 130)
                    }
                    HStack(spacing: 10) {
                        Text("Sampling")
                            .frame(width: 130, alignment: .trailing)
                        Picker("Sampling", selection: $config.samplingMode) {
                            ForEach(LocalDatasetClipSamplingMode.allCases) { mode in
                                Text(mode.displayName).tag(mode)
                            }
                        }
                        .labelsHidden()
                        .pickerStyle(.segmented)
                        .disabled(isBusy)
                        .frame(width: 300)
                    }
                    Text(config.samplingMode.helpText)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .frame(maxWidth: .infinity, alignment: .leading)
                    if config.samplingMode == .randomSeeded {
                        LabeledContent("Random Seed") {
                            TextField("1337", value: $config.samplingSeed, format: .number)
                                .multilineTextAlignment(.trailing)
                                .monospacedDigit()
                                .disabled(isBusy)
                                .frame(width: 130)
                                .onChange(of: config.samplingSeed) { _, newValue in
                                    config.samplingSeed = min(max(newValue, 0), 999_999)
                                }
                        }
                    }
                    LabeledContent("Max Total Clips") {
                        Stepper(
                            "\(config.maxTotalClips)",
                            value: $config.maxTotalClips,
                            in: 5 ... LocalFilesDatasetConfig.maximumTotalClips,
                            step: 5
                        )
                        .disabled(isBusy)
                        .frame(maxWidth: 130)
                    }
                    LabeledContent("Clip Format") {
                        Picker("Clip Format", selection: $config.clipFormat) {
                            ForEach(LocalDatasetClipFormat.allCases) { format in
                                Text(format.displayName).tag(format)
                            }
                        }
                        .labelsHidden()
                        .pickerStyle(.segmented)
                        .disabled(isBusy)
                        .frame(maxWidth: 160)
                    }
                    Text(config.clipFormat.helpText)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                datasetCoverageSummary
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

    private var datasetCoverageSummary: some View {
        let summary = coverageSummary

        return HStack(alignment: .firstTextBaseline, spacing: 6) {
            Image(systemName: summary.isCapped ? "exclamationmark.triangle.fill" : "checkmark.circle.fill")
                .foregroundStyle(summary.isCapped ? .orange : .green)
            Text(summary.message)
                .font(.caption)
                .foregroundStyle(summary.isCapped ? .orange : .secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var coverageSummary: (message: String, isCapped: Bool) {
        let fileCount = selectedFiles.count
        guard fileCount > 0 else {
            return ("Add source files to preview dataset coverage.", false)
        }
        if config.buildMode == .wholeTracks {
            return (
                "Will preserve all \(fileCount) complete source track\(fileCount == 1 ? "" : "s"). Full mixes are selected by default; stems remain opt-in.",
                false
            )
        }

        let requestedClips = fileCount * config.clipsPerTrack
        let plannedClips = min(requestedClips, config.maxTotalClips)
        if requestedClips <= config.maxTotalClips {
            return (
                "Will create up to \(plannedClips) clip\(plannedClips == 1 ? "" : "s") across all \(fileCount) file\(fileCount == 1 ? "" : "s").",
                false
            )
        }

        let fullFiles = config.maxTotalClips / max(1, config.clipsPerTrack)
        let partialFileCount = config.maxTotalClips % max(1, config.clipsPerTrack) == 0 ? 0 : 1
        let reachedFiles = min(fileCount, fullFiles + partialFileCount)
        return (
            "Max Total Clips will create \(plannedClips) clips and stop after about \(reachedFiles) of \(fileCount) files. Raise the cap or lower Clips per File for full coverage.",
            true
        )
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
        case let .chunking(idx, total, _):
            if idx == 0 {
                return "Preparing files..."
            }
            return config.buildMode == .wholeTracks
                ? "Preserving track \(idx) of \(total)"
                : "Chunking \(idx) of \(total)"
        default:
            return ""
        }
    }

    private var progressDetail: String {
        switch builder.phase {
        case let .chunking(_, _, name): name
        default: ""
        }
    }

    @ViewBuilder
    private var progressIndicator: some View {
        switch builder.phase {
        case let .chunking(idx, total, _):
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
                Text(
                    config.buildMode == .wholeTracks
                        ? "\(editableClips.count) whole tracks packaged for training"
                        : "\(editableClips.count) clips packaged for training"
                )
                .font(.callout)
                .fontWeight(.medium)
                Spacer()
                if case let .complete(dir, _) = builder.phase {
                    Button {
                        NSWorkspace.shared.open(dir)
                    } label: {
                        Label("Show in Finder", systemImage: "folder")
                            .font(.caption)
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                    .disabled(enrichment.phase.isRunning)
                    Button {
                        exportPackage(from: dir)
                    } label: {
                        Label("Export Package...", systemImage: "archivebox")
                            .font(.caption)
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                    .disabled(isExportingPackage || enrichment.phase.isRunning)
                    Menu {
                        Picker(
                            "Annotation Backend",
                            selection: $enrichmentOptions.annotationBackend
                        ) {
                            ForEach(
                                LocalDatasetAnnotationBackendChoice.allCases
                            ) {
                                Text($0.title).tag($0)
                            }
                        }
                        Toggle(
                            "Create vocal and instrumental stems",
                            isOn: $enrichmentOptions.createStems
                        )
                    } label: {
                        Label("Options", systemImage: "slider.horizontal.3")
                            .font(.caption)
                    }
                    .menuStyle(.borderlessButton)
                    .fixedSize()
                    .disabled(enrichment.phase.isRunning || isExportingPackage)
                    Button {
                        startEnrichment(from: dir)
                    } label: {
                        Label("Enrich Dataset", systemImage: "waveform.path.ecg")
                            .font(.caption)
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.small)
                    .disabled(enrichment.phase.isRunning || isExportingPackage)
                }
            }
            .padding(10)
            .background(.green.opacity(0.08))
            .clipShape(RoundedRectangle(cornerRadius: 8))

            enrichmentStatusSection

            if let packageSummary {
                Text(packageSummary)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Text("Preview captions below. Open the dataset review workspace to approve clips and launch training.")
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
                            set: {
                                editableClips[idx].description = $0
                                hasUnsavedCaptionEdits = true
                            }
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
                .disabled(isBusy)

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
                if enrichment.phase.isRunning {
                    Button("Stop Enrichment", role: .destructive) {
                        enrichment.cancel()
                    }
                    .buttonStyle(.bordered)
                } else {
                    Button("Review Dataset") {
                        openDatasetReview()
                    }
                    .buttonStyle(.borderedProminent)
                }
            }
        }
        .padding(16)
    }

    @ViewBuilder
    private var enrichmentStatusSection: some View {
        switch enrichment.phase {
        case .idle:
            EmptyView()
        case let .running(step):
            HStack(spacing: 10) {
                ProgressView()
                    .controlSize(.small)
                VStack(alignment: .leading, spacing: 2) {
                    Text(step.title)
                        .font(.caption)
                        .fontWeight(.medium)
                    Text("This may take a while for larger clip sets.")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
                Spacer()
            }
            .padding(10)
            .background(.blue.opacity(0.08))
            .clipShape(RoundedRectangle(cornerRadius: 8))
        case let .complete(result):
            HStack(alignment: .top, spacing: 8) {
                Image(systemName: "checkmark.circle.fill")
                    .foregroundStyle(.green)
                    .padding(.top, 1)
                Text(result.summary)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Spacer()
            }
            .padding(10)
            .background(.green.opacity(0.08))
            .clipShape(RoundedRectangle(cornerRadius: 8))
        case let .failed(message):
            HStack(alignment: .top, spacing: 8) {
                Image(systemName: "exclamationmark.triangle.fill")
                    .foregroundStyle(.orange)
                    .padding(.top, 1)
                Text(message)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
                Spacer()
            }
            .padding(10)
            .background(.orange.opacity(0.08))
            .clipShape(RoundedRectangle(cornerRadius: 8))
        }
    }

    private var isBusy: Bool {
        builder.phase.isRunning || enrichment.phase.isRunning
    }

    private var canStart: Bool {
        !selectedFiles.isEmpty && outputManager.directory != nil && !isBusy
    }

    private func startImport() {
        guard let outputBase = outputManager.directory else { return }
        let template = styleTemplate.trimmingCharacters(in: .whitespacesAndNewlines)
        builder.start(
            sourceFiles: selectedFiles,
            sourceDirectories: selectedFolders,
            config: config,
            outputBase: outputBase,
            datasetName: datasetName,
            styleTemplate: template
        )
    }

    @discardableResult
    private func persistCaptionsIfNeeded() -> Bool {
        guard case let .complete(dir, _) = builder.phase else { return false }
        guard hasUnsavedCaptionEdits else { return true }
        do {
            try TrainingDatasetPackageWriter.writeLocalPackage(
                outputDirectory: dir,
                clips: editableClips,
                sourceFiles: selectedFiles,
                config: config,
                styleTemplate: styleTemplate
            )
            hasUnsavedCaptionEdits = false
            reloadEditableClips(from: dir, fallback: editableClips)
            refreshPackageSummary(for: dir)
            return true
        } catch {
            sourceNotice = error.localizedDescription
            return false
        }
    }

    private func startEnrichment(from dir: URL) {
        guard persistCaptionsIfNeeded() else { return }
        enrichment.enrich(
            datasetDirectory: dir,
            options: enrichmentOptions,
            prepareMemory: { [generationService] in
                _ = await generationService.unloadAllWeights()
            }
        )
    }

    private func openDatasetReview() {
        guard case let .complete(dir, _) = builder.phase else { return }
        guard persistCaptionsIfNeeded() else { return }
        onOpenDatasetReview(dir)
        dismiss()
    }

    private func refreshPackageSummary(for dir: URL) {
        packageSummary = TrainingDatasetPackageWriter.validatePackage(at: dir).summary
    }

    private func exportPackage(from dir: URL) {
        guard persistCaptionsIfNeeded() else { return }

        let panel = NSSavePanel()
        panel.allowedContentTypes = [.zip]
        panel.canCreateDirectories = true
        panel.nameFieldStringValue = "\(dir.lastPathComponent).zip"
        guard panel.runModal() == .OK, let destination = panel.url else { return }

        isExportingPackage = true
        Task {
            defer { isExportingPackage = false }
            do {
                let archiveURL = try await TrainingDatasetPackageWriter.exportPackageArchive(
                    outputDirectory: dir,
                    to: destination
                )
                sourceNotice = "Package exported to \(archiveURL.lastPathComponent)."
            } catch is CancellationError {
                sourceNotice = "Package export was cancelled."
            } catch {
                sourceNotice = error.localizedDescription
            }
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
        let audioURLs = urls
            .filter { LocalAudioFileDiscovery.supportedExtensions.contains($0.pathExtension.lowercased()) }
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
        var seen = Set(existing.map(\.standardizedFileURL.path))
        var merged = existing
        for url in incoming where seen.insert(url.standardizedFileURL.path).inserted {
            merged.append(url)
        }
        return merged
    }

    private func reloadEditableClips(from dir: URL, fallback: [TrainingImportedFile] = []) {
        do {
            editableClips = try TrainingDatasetImporter.importFiles(
                from: dir.appendingPathComponent(TrainingDatasetPackageWriter.fullMixBundleFileName)
            )
        } catch {
            editableClips = fallback
            sourceNotice = "Could not reload package captions: \(error.localizedDescription)"
        }
    }
}
