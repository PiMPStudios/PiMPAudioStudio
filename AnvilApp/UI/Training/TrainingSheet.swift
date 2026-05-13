import SwiftUI
import AnvilCore

struct TrainingSheet: View {
    @EnvironmentObject private var trainingService: TrainingService
    @EnvironmentObject private var historyStore: HistoryStore

    let selectedEntries: [HistoryEntry]

    @Environment(\.dismiss) private var dismiss

    @State private var adapterName: String = ""
    private let resumeAdapterID: String?
    @State private var selectedRank: Int = 16
    @State private var selectedEpochs: Int = 10
    @State private var selectedTargetLayers: Set<TargetLayer> = [.selfAttention, .crossAttention]
    @State private var warmupFraction: Float = 0.10
    @State private var gradientAccumulationSteps: Int = 1
    @State private var importedFiles: [(url: URL, description: String)]
    @State private var showingFileImporter: Bool = false
    @State private var showingStopConfirmation: Bool = false

    init(
        selectedEntries: [HistoryEntry],
        initialImportedFiles: [(url: URL, description: String)] = [],
        initialName: String = "",
        initialConfig: LoRAConfig? = nil,
        resumeAdapterID: String? = nil
    ) {
        self.selectedEntries = selectedEntries
        self.resumeAdapterID = resumeAdapterID
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

    private let rankOptions = [8, 16, 32]
    private let epochOptions = [5, 10, 20, 50]
    private let accumOptions = [1, 2, 4, 8]

    var body: some View {
        VStack(spacing: 0) {
            // Header
            HStack {
                Text("Train LoRA Adapter")
                    .font(.headline)
                Spacer()
                Button("Close") { dismiss() }
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
        .frame(minWidth: 500, minHeight: 500)
        .onAppear {
            trainingService.resetToIdle()
        }
        .fileImporter(
            isPresented: $showingFileImporter,
            allowedContentTypes: [.audio],
            allowsMultipleSelection: true
        ) { result in
            if case .success(let urls) = result {
                for url in urls {
                    importedFiles.append((url: url, description: ""))
                }
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
        VStack(alignment: .leading, spacing: 16) {
            // Resume banner
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

            // Training data section
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
                        HStack {
                            Image(systemName: "doc.badge.plus")
                                .foregroundStyle(.secondary)
                            TextField("Description", text: Binding(
                                get: { importedFiles[idx].description },
                                set: { importedFiles[idx].description = $0 }
                            ))
                            .font(.callout)
                            .textFieldStyle(.roundedBorder)
                            .disabled(isCheckpointResume)
                            Button {
                                importedFiles.remove(at: idx)
                            } label: {
                                Image(systemName: "xmark.circle")
                                    .foregroundStyle(.secondary)
                            }
                            .buttonStyle(.borderless)
                            .disabled(isCheckpointResume)
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
                    Button("+ Add audio files...") {
                        showingFileImporter = true
                    }
                    .buttonStyle(.borderless)
                    .font(.caption)
                    .disabled(isCheckpointResume)
                }
            }

            // Configuration section
            VStack(alignment: .leading, spacing: 8) {
                Text("Configuration")
                    .font(.subheadline)
                    .fontWeight(.semibold)

                LabeledContent("Adapter Name") {
                    TextField("e.g. My Jazz Style", text: $adapterName)
                        .textFieldStyle(.roundedBorder)
                        .frame(maxWidth: 200)
                        .disabled(isCheckpointResume)
                }
                LabeledContent("Base Model") {
                    Text("ACE-Step DiT (active)")
                        .foregroundStyle(.secondary)
                }
                LabeledContent("Rank") {
                    Picker("Rank", selection: $selectedRank) {
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
                    .frame(maxWidth: 200)
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

            // Error display
            if case .failed(let message) = trainingService.state {
                Label(message, systemImage: "exclamationmark.triangle")
                    .foregroundStyle(.red)
                    .font(.caption)
            }

            // Start/Resume button
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
        .padding()
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
                Text("Rank \(adapter.config.rank) \u{2022} \(adapter.sampleCount) tracks")
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

                Button("Done") { dismiss() }
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
        trainingService.startTraining(
            entries: selectedEntries,
            audioURLs: audioURLs,
            importedFiles: importedFiles,
            adapterName: adapterName,
            config: config,
            resumeAdapterID: resumeAdapterID
        )
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
