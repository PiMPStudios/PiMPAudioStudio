import SwiftUI
import PiMPAudioCore
import UniformTypeIdentifiers

/// Drives `.sheet(item:)` for any training sheet presentation.
/// Carrying all data as value types ensures the sheet closure sees the
/// correct dataset regardless of when SwiftUI evaluates it.
struct TrainingPresentation: Identifiable {
    let id = UUID()
    /// History entries resolved from the manifest (may be empty for new sessions).
    let selectedEntries: [HistoryEntry]
    /// Imported file entries — from manifest, YouTube, or manual selection.
    let files: [TrainingImportedFile]
    /// Pre-fill the adapter name field (empty = blank for new sessions).
    let initialName: String
    /// Pre-fill training config (nil = use defaults for new sessions).
    let initialConfig: LoRAConfig?
    /// Adapter implementation for this training session.
    let adapterKind: AdapterKind
    /// Adapter ID to resume from a saved checkpoint. Nil means fresh training/retraining.
    let resumeAdapterID: String?

    /// Blank new training session.
    init(files: [TrainingImportedFile] = [], adapterKind: AdapterKind = .lora, initialName: String = "") {
        self.selectedEntries = []
        self.files = files
        self.initialName = initialName
        self.initialConfig = nil
        self.adapterKind = adapterKind
        self.resumeAdapterID = nil
    }

    /// Pre-populated retrain session.
    init(
        selectedEntries: [HistoryEntry],
        files: [TrainingImportedFile],
        initialName: String,
        initialConfig: LoRAConfig?,
        adapterKind: AdapterKind = .lora,
        resumeAdapterID: String? = nil
    ) {
        self.selectedEntries = selectedEntries
        self.files = files
        self.initialName = initialName
        self.initialConfig = initialConfig
        self.adapterKind = adapterKind
        self.resumeAdapterID = resumeAdapterID
    }
}

enum TrainingSurfacePresentationStyle {
    case sheet
    case inline
}

struct TrainingSheet: View {
    @EnvironmentObject private var trainingService: TrainingService
    @EnvironmentObject private var historyStore: HistoryStore

    let selectedEntries: [HistoryEntry]

    @Environment(\.dismiss) private var dismiss

    private let presentationStyle: TrainingSurfacePresentationStyle
    private let onClose: (() -> Void)?
    private let adapterKind: AdapterKind

    @State private var adapterName: String = ""
    private let resumeAdapterID: String?
    @State private var selectedRank: Int = 16
    @State private var selectedEpochs: Int = 10
    @State private var selectedTargetLayers: Set<TargetLayer> = [.selfAttention, .crossAttention]
    @State private var warmupFraction: Float = 0.10
    @State private var gradientAccumulationSteps: Int = 1
    @State private var importedFiles: [TrainingImportedFile]
    @State private var showingFileImporter: Bool = false
    @State private var showingStopConfirmation: Bool = false
    @State private var baseModelDisplayName: String = "ACE-Step DiT"
    @State private var importErrorMessage: String?
    @State private var importStatusMessage: String?

    init(
        selectedEntries: [HistoryEntry],
        initialImportedFiles: [TrainingImportedFile] = [],
        initialName: String = "",
        initialConfig: LoRAConfig? = nil,
        adapterKind: AdapterKind = .lora,
        resumeAdapterID: String? = nil,
        presentationStyle: TrainingSurfacePresentationStyle = .sheet,
        onClose: (() -> Void)? = nil
    ) {
        self.selectedEntries = selectedEntries
        self.resumeAdapterID = resumeAdapterID
        self.presentationStyle = presentationStyle
        self.onClose = onClose
        self.adapterKind = adapterKind
        self._importedFiles = State(initialValue: initialImportedFiles)
        self._adapterName = State(initialValue: initialName)
        if let config = initialConfig {
            self._selectedRank = State(initialValue: config.rank)
            self._selectedEpochs = State(initialValue: config.maxEpochs)
            self._selectedTargetLayers = State(initialValue: config.targetLayers)
            self._warmupFraction = State(initialValue: config.warmupFraction)
            self._gradientAccumulationSteps = State(initialValue: config.gradientAccumulationSteps)
        }
    }

    private var rankOptions: [Int] {
        adapterKind == .lokr ? [16, 32, 64, 128] : [8, 16, 32, 64]
    }
    private let epochOptions = [5, 10, 20, 50, 75, 100]
    private let accumOptions = [1, 2, 4, 8]

    var body: some View {
        VStack(spacing: 0) {
            // Header
            HStack {
                Text("Train \(adapterKind.title) Adapter")
                    .font(.headline)
                Spacer()
                Button("Close") { close() }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
            }
            .padding()

            Divider()

            // Content based on state
            ScrollView {
                switch trainingService.state {
                case .idle, .failed, .cancelled:
                    configView
                case .paused:
                    configView
                case .loadingModels, .preparingDataset, .training, .saving:
                    progressView
                case .complete(let adapter):
                    completionView(adapter: adapter)
                }
            }
        }
        .frame(
            minWidth: presentationStyle == .sheet ? 500 : 0,
            minHeight: presentationStyle == .sheet ? 500 : 0
        )
        .frame(
            maxWidth: presentationStyle == .inline ? .infinity : nil,
            maxHeight: presentationStyle == .inline ? .infinity : nil,
            alignment: .topLeading
        )
        .onAppear {
            trainingService.resetToIdle()
        }
        .task {
            await refreshBaseModelDisplayName()
        }
        .onReceive(NotificationCenter.default.publisher(for: .modelRegistryDidChange)) { _ in
            Task { await refreshBaseModelDisplayName() }
        }
        .fileImporter(
            isPresented: $showingFileImporter,
            allowedContentTypes: [.audio, .json],
            allowsMultipleSelection: true
        ) { result in
            if case .success(let urls) = result {
                importSelectedFiles(urls)
            } else if case .failure(let error) = result {
                importErrorMessage = error.localizedDescription
            }
        }
        .alert("Pause or Stop Training?", isPresented: $showingStopConfirmation) {
            Button("Continue Training", role: .cancel) {}
            if trainingService.canSaveOnStop {
                Button("Save & Pause") {
                    trainingService.stopTraining(saveCheckpoint: true)
                }
            }
            Button("Discard & Stop", role: .destructive) {
                trainingService.stopTraining(saveCheckpoint: false)
            }
        } message: {
            if trainingService.canSaveOnStop {
                Text("Training has made progress. Save a checkpoint so you can resume this adapter later.")
            } else {
                Text("No training step has completed yet. Progress will be lost.")
            }
        }
    }

    // MARK: - Config View

    private var configView: some View {
        Group {
            if presentationStyle == .inline {
                inlineConfigView
            } else {
                sheetConfigView
            }
        }
    }

    private var sheetConfigView: some View {
        VStack(alignment: .leading, spacing: 16) {
            resumeBanner
            trainingDataSection
            configurationSection
            errorDisplay
            startButtonRow
        }
        .padding()
    }

    private var inlineConfigView: some View {
        VStack(alignment: .leading, spacing: 20) {
            resumeBanner

            HStack(alignment: .top, spacing: 32) {
                trainingDataSection
                    .frame(minWidth: 280, idealWidth: 360, maxWidth: 420, alignment: .topLeading)

                VStack(alignment: .leading, spacing: 16) {
                    configurationSection
                    errorDisplay
                    startButtonRow
                }
                .frame(minWidth: 360, idealWidth: 460, maxWidth: 560, alignment: .topLeading)
            }
            .frame(maxWidth: .infinity, alignment: .topLeading)
        }
        .padding(24)
    }

    @ViewBuilder
    private var resumeBanner: some View {
        if isCheckpointResume {
            HStack {
                Image(systemName: "pause.circle.fill")
                    .foregroundStyle(.orange)
                Text("Training paused. Resume from the saved checkpoint, or start fresh from Retrain.")
                    .font(.callout)
                Spacer()
            }
            .padding(8)
            .background(.orange.opacity(0.1))
            .clipShape(RoundedRectangle(cornerRadius: 8))
        }
    }

    private var trainingDataSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Training Data")
                .font(.subheadline)
                .fontWeight(.semibold)

            VStack(spacing: 4) {
                ForEach(selectedEntries) { entry in
                    HStack {
                        Image(systemName: "waveform")
                            .foregroundStyle(.secondary)
                        Text(entry.prompt)
                            .lineLimit(1)
                            .font(.callout)
                        Spacer()
                        Text("\(Int(entry.duration))s")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
                ForEach(importedFiles.indices, id: \.self) { idx in
                    HStack(alignment: .top, spacing: 8) {
                        Image(systemName: "doc.badge.plus")
                            .foregroundStyle(.secondary)
                            .padding(.top, 6)
                        VStack(alignment: .leading, spacing: 6) {
                            Text(importedFiles[idx].url.lastPathComponent)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                                .lineLimit(1)
                                .truncationMode(.middle)
                            TextField("Description", text: Binding(
                                get: { importedFiles[idx].description },
                                set: { importedFiles[idx].description = $0 }
                            ))
                            .font(.callout)
                            .textFieldStyle(.roundedBorder)
                            .disabled(isCheckpointResume)
                            if !importedFiles[idx].lyrics.isEmpty {
                                TextField("Lyrics", text: Binding(
                                    get: { importedFiles[idx].lyrics },
                                    set: { importedFiles[idx].lyrics = $0 }
                                ))
                                .font(.caption)
                                .textFieldStyle(.roundedBorder)
                                .disabled(isCheckpointResume)
                            }
                        }
                        Button {
                            importedFiles.remove(at: idx)
                        } label: {
                            Image(systemName: "xmark.circle")
                                .foregroundStyle(.secondary)
                        }
                        .buttonStyle(.borderless)
                        .disabled(isCheckpointResume)
                        .padding(.top, 6)
                    }
                }
            }
            .padding(8)
            .background(.quaternary.opacity(0.3))
            .clipShape(RoundedRectangle(cornerRadius: 8))

            HStack {
                let totalCount = selectedEntries.count + importedFiles.count
                Text("\(totalCount) tracks selected")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Spacer()
                Button("+ Add audio files or JSON...") {
                    showingFileImporter = true
                }
                .buttonStyle(.borderless)
                .font(.caption)
                .disabled(isCheckpointResume)
            }
            if let importStatusMessage {
                Label(importStatusMessage, systemImage: "checkmark.circle")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            if let importErrorMessage {
                Label(importErrorMessage, systemImage: "exclamationmark.triangle")
                    .font(.caption)
                    .foregroundStyle(.orange)
            }
        }
    }

    private var configurationSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Configuration")
                .font(.subheadline)
                .fontWeight(.semibold)

            LabeledContent("Adapter Name") {
                TextField("e.g. My Jazz Style", text: $adapterName)
                    .textFieldStyle(.roundedBorder)
                    .frame(maxWidth: 240)
                    .disabled(isCheckpointResume)
            }
            LabeledContent("Base Model") {
                Text(baseModelDisplayName)
                    .foregroundStyle(.secondary)
            }
            LabeledContent(adapterKind == .lokr ? "Linear Dim" : "Rank") {
                Picker(adapterKind == .lokr ? "Linear Dim" : "Rank", selection: $selectedRank) {
                    ForEach(rankOptions, id: \.self) { Text("\($0)").tag($0) }
                }
                .pickerStyle(.segmented)
                .frame(maxWidth: 150)
                .disabled(isCheckpointResume)
            }
            LabeledContent("Epochs") {
                Picker("Epochs", selection: $selectedEpochs) {
                    ForEach(epochOptions, id: \.self) { Text("\($0)").tag($0) }
                }
                .pickerStyle(.segmented)
                .frame(maxWidth: 220)
                .disabled(isCheckpointResume)
            }

            // Target layers
            LabeledContent("Target Layers") {
                HStack(spacing: 8) {
                    ForEach(TargetLayer.allCases, id: \.self) { layer in
                        Toggle(layer.displayName, isOn: Binding(
                            get: { selectedTargetLayers.contains(layer) },
                            set: { isOn in
                                if isOn {
                                    selectedTargetLayers.insert(layer)
                                } else if selectedTargetLayers.count > 1 {
                                    selectedTargetLayers.remove(layer)
                                }
                            }
                        ))
                        .toggleStyle(.checkbox)
                        .disabled(isCheckpointResume)
                    }
                }
            }

            // Advanced: warmup + accumulation
            DisclosureGroup("Advanced") {
                LabeledContent("LR Warmup") {
                    HStack {
                        Slider(
                            value: Binding(
                                get: { Double(warmupFraction) },
                                set: { warmupFraction = Float($0) }
                            ),
                            in: 0...0.5,
                            step: 0.05
                        )
                        .frame(maxWidth: 150)
                        .disabled(isCheckpointResume)
                        Text("\(Int(warmupFraction * 100))%")
                            .monospacedDigit()
                            .frame(width: 40)
                    }
                }
                LabeledContent("Grad. Accumulation") {
                    Picker("Steps", selection: $gradientAccumulationSteps) {
                        ForEach(accumOptions, id: \.self) { Text("\($0)").tag($0) }
                    }
                    .pickerStyle(.segmented)
                    .frame(maxWidth: 200)
                    .disabled(isCheckpointResume)
                }
            }
        }
    }

    @ViewBuilder
    private var errorDisplay: some View {
        if case .failed(let message) = trainingService.state {
            Label(message, systemImage: "exclamationmark.triangle")
                .foregroundStyle(.red)
                .font(.caption)
        }
    }

    private var startButtonRow: some View {
        HStack {
            Spacer()
            if isCheckpointResume {
                Button("Resume Training") {
                    startTraining()
                }
                .buttonStyle(.borderedProminent)
            } else {
                Button("Start Training") {
                    startTraining()
                }
                .buttonStyle(.borderedProminent)
                .disabled(!canStartTraining)
            }
        }
    }

    // MARK: - Progress View

    private var progressView: some View {
        VStack(spacing: 16) {
            switch trainingService.state {
            case .loadingModels:
                VStack(spacing: 8) {
                    ProgressView()
                        .controlSize(.large)
                    Text("Loading ACE-Step Models")
                        .font(.headline)
                    Text("Preparing DiT, VAE, tokenizer, text encoders, and silence prior for training.")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                }
                .padding(.top, 40)

            case .preparingDataset(let current, let total):
                VStack(spacing: 8) {
                    Text("Preparing Dataset")
                        .font(.headline)
                    Text("Encoding sample \(current) of \(total)")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                    ProgressView(value: Double(current), total: Double(total))
                        .frame(maxWidth: 300)
                }
                .padding(.top, 40)

            case .training(let uiState):
                VStack(spacing: 12) {
                    Text("Training \"\(adapterName)\"")
                        .font(.headline)
                    Text("Epoch \(uiState.epoch) of \(uiState.totalEpochs)")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)

                    ProgressView(value: Double(uiState.epoch), total: Double(uiState.totalEpochs))
                        .frame(maxWidth: 300)

                    // Loss curve
                    LossCurveView(lossHistory: uiState.lossHistory)
                        .padding(.horizontal)

                    // Stats row
                    HStack(spacing: 20) {
                        statView(value: String(format: "%.4f", uiState.loss), label: "Loss")
                        statView(
                            value: String(format: "%.1e", uiState.learningRate),
                            label: "LR"
                        )
                        statView(value: formatTime(uiState.elapsed), label: "Elapsed")
                        if let remaining = uiState.remaining {
                            statView(value: "~\(formatTime(remaining))", label: "ETA")
                        }
                    }
                }
                .padding(.top, 20)

            case .saving:
                VStack(spacing: 8) {
                    ProgressView()
                    Text("Saving adapter...")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
                .padding(.top, 40)

            default:
                EmptyView()
            }

            Spacer()

            if case .training = trainingService.state {
                Button("Pause / Stop Training", role: .destructive) {
                    showingStopConfirmation = true
                }
                .buttonStyle(.bordered)
            }
        }
        .padding()
    }

    // MARK: - Completion View

    private func completionView(adapter: LoRAAdapter) -> some View {
        VStack(spacing: 16) {
            Spacer()

            Image(systemName: "checkmark.circle.fill")
                .font(.largeTitle)
                .foregroundStyle(.green)

            Text("Training Complete")
                .font(.headline)

            VStack(spacing: 4) {
                Text(adapter.name)
                    .font(.title3)
                    .fontWeight(.semibold)
                if let loss = adapter.finalLoss {
                    Text("Final loss: \(String(format: "%.4f", loss))")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Text("\(adapter.kind == .lokr ? "Linear dim" : "Rank") \(adapter.kind == .lokr ? adapter.lokrConfig?.linearDim ?? adapter.config.rank : adapter.config.rank) \u{2022} \(adapter.sampleCount) tracks")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Spacer()

            HStack(spacing: 12) {
                Button("Train Again") {
                    trainingService.resetToIdle()
                }
                .buttonStyle(.bordered)
                .help("Return to config with the same dataset — adjust epochs or settings and retrain")

                Button("Done") { close() }
                    .buttonStyle(.borderedProminent)
            }
        }
        .padding()
    }

    // MARK: - Helpers

    private var canStartTraining: Bool {
        !adapterName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && !(selectedEntries.isEmpty && importedFiles.isEmpty)
            && !selectedTargetLayers.isEmpty
    }

    private var isResumeSession: Bool {
        resumeAdapterID != nil
    }

    private var isPausedState: Bool {
        if case .paused = trainingService.state { return true }
        return false
    }

    private var isCheckpointResume: Bool {
        isResumeSession || isPausedState
    }

    private func close() {
        if let onClose {
            onClose()
        } else {
            dismiss()
        }
    }

    private func statView(value: String, label: String) -> some View {
        VStack(spacing: 2) {
            Text(value)
                .font(.title3)
                .fontWeight(.semibold)
                .monospacedDigit()
            Text(label)
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
    }

    private func formatTime(_ interval: TimeInterval) -> String {
        let minutes = Int(interval) / 60
        let seconds = Int(interval) % 60
        return "\(minutes):\(String(format: "%02d", seconds))"
    }

    private func startTraining() {
        let audioURLs = selectedEntries.map { historyStore.fileURL(for: $0) }
        let config = LoRAConfig(
            rank: selectedRank,
            targetLayers: selectedTargetLayers,
            maxEpochs: selectedEpochs,
            warmupFraction: warmupFraction,
            gradientAccumulationSteps: gradientAccumulationSteps
        )
        switch adapterKind {
        case .lora:
            trainingService.startTraining(
                entries: selectedEntries,
                audioURLs: audioURLs,
                importedFiles: importedFiles,
                adapterName: adapterName,
                config: config,
                resumeAdapterID: resumeAdapterID
            )
        case .lokr:
            trainingService.startLoKRTraining(
                entries: selectedEntries,
                audioURLs: audioURLs,
                importedFiles: importedFiles,
                adapterName: adapterName,
                config: LoKRConfig(
                    linearDim: selectedRank,
                    alpha: Float(selectedRank),
                    targetLayers: selectedTargetLayers,
                    maxEpochs: selectedEpochs,
                    warmupFraction: warmupFraction,
                    gradientAccumulationSteps: gradientAccumulationSteps
                )
            )
        }
    }

    private func refreshBaseModelDisplayName() async {
        let name = await trainingService.activeBaseModelDisplayName()
        await MainActor.run {
            baseModelDisplayName = name
        }
    }

    private func importSelectedFiles(_ urls: [URL]) {
        importErrorMessage = nil
        importStatusMessage = nil
        var importedJSONCount = 0
        var importedAudioCount = 0
        var lyricCount = 0
        var duplicateCount = 0

        for url in urls {
            if url.pathExtension.lowercased() == "json" {
                do {
                    let imported = try TrainingDatasetImporter.importFiles(from: url)
                    let result = appendImportedFiles(imported)
                    importedJSONCount += result.added
                    duplicateCount += result.duplicates
                    lyricCount += imported.filter { !$0.lyrics.isEmpty }.count
                } catch {
                    importErrorMessage = error.localizedDescription
                }
            } else {
                let result = appendImportedFiles([TrainingImportedFile(url: url, description: "")])
                importedAudioCount += result.added
                duplicateCount += result.duplicates
            }
        }

        importStatusMessage = importSummary(
            jsonCount: importedJSONCount,
            audioCount: importedAudioCount,
            lyricCount: lyricCount,
            duplicateCount: duplicateCount
        )
    }

    private func appendImportedFiles(_ files: [TrainingImportedFile]) -> (added: Int, duplicates: Int) {
        var seen = Set(importedFiles.map { $0.url.standardizedFileURL.path })
        var added = 0
        var duplicates = 0

        for file in files {
            if seen.insert(file.url.standardizedFileURL.path).inserted {
                importedFiles.append(file)
                added += 1
            } else {
                duplicates += 1
            }
        }

        return (added, duplicates)
    }

    private func importSummary(
        jsonCount: Int,
        audioCount: Int,
        lyricCount: Int,
        duplicateCount: Int
    ) -> String? {
        guard jsonCount > 0 || audioCount > 0 || duplicateCount > 0 else { return nil }

        var parts: [String] = []
        if jsonCount > 0 {
            let lyricSuffix = lyricCount > 0 ? ", \(lyricCount) with lyrics" : ""
            parts.append("Imported \(jsonCount) from JSON\(lyricSuffix)")
        }
        if audioCount > 0 {
            parts.append("Added \(audioCount) audio file\(audioCount == 1 ? "" : "s")")
        }
        if duplicateCount > 0 {
            parts.append("Skipped \(duplicateCount) duplicate\(duplicateCount == 1 ? "" : "s")")
        }
        return parts.joined(separator: ". ")
    }
}

private extension AdapterKind {
    var title: String {
        switch self {
        case .lora: return "LoRA"
        case .lokr: return "LoKR"
        }
    }
}

// MARK: - TargetLayer Display

extension TargetLayer {
    var displayName: String {
        switch self {
        case .selfAttention: return "Self-Attn"
        case .crossAttention: return "Cross-Attn"
        case .mlp: return "MLP"
        }
    }
}
