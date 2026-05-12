import SwiftUI
import AVFoundation
import AnvilCore

struct EditorView: View {
    @StateObject private var state = EditorState()
    @EnvironmentObject private var editorService: EditorService

    var preloadedAudioURL: URL?
    var preloadedPrompt: String?

    var body: some View {
        VStack(spacing: 0) {
            EditorToolbar(
                state: state,
                onOpen: openFile,
                onExport: exportFile,
                onUndo: performUndo,
                onRedo: performRedo
            )

            if state.isLoaded {
                ScrollView {
                    VStack(spacing: 16) {
                        waveformSection
                        if state.editMode == .inpaint {
                            frequencySection
                        }
                        EditorPromptArea(state: state)
                        EditorControlsBar(state: state, onExecute: executeEdit)

                        Divider()
                            .padding(.horizontal, 16)

                        DSPPaletteView(state: state) { effect, params in
                            state.applyDSPEffect(effect, parameters: params)
                        }
                    }
                    .padding(.vertical, 16)
                }
            } else {
                emptyState
            }
        }
        .onAppear {
            if let url = preloadedAudioURL {
                loadFromURL(url)
            }
            if let prompt = preloadedPrompt {
                state.originalPrompt = prompt
            }
        }
        .background {
            Group {
                Button("") {
                    if state.hasSelection && state.isLoaded {
                        state.applyDSPEffect(.trimToSelection)
                    }
                }
                .keyboardShortcut("t", modifiers: .command)

                Button("") {
                    if state.hasSelection && state.isLoaded {
                        state.applyDSPEffect(.deleteSelection)
                    }
                }
                .keyboardShortcut(.delete, modifiers: .command)

                Button("") {
                    if state.isLoaded {
                        state.applyDSPEffect(.normalize)
                    }
                }
                .keyboardShortcut("n", modifiers: [.command, .shift])

                Button("") {
                    if state.isLoaded {
                        withAnimation { state.expandedEffect = .eq }
                    }
                }
                .keyboardShortcut("e", modifiers: [.command, .shift])

                Button("") {
                    if state.isLoaded {
                        withAnimation { state.expandedEffect = .reverb }
                    }
                }
                .keyboardShortcut("r", modifiers: [.command, .shift])
            }
            .frame(width: 0, height: 0)
            .opacity(0)
        }
    }

    private var waveformSection: some View {
        VStack(spacing: 8) {
            EditorWaveformView(
                peaks: state.waveformPeaks,
                duration: state.duration,
                playbackProgress: state.playbackProgress,
                selectionStart: $state.selectionStart,
                selectionEnd: $state.selectionEnd,
                isSelectionEnabled: state.editMode == .inpaint
            )
            .padding(.horizontal, 16)

            HStack {
                Text(formatTime(0))
                Spacer()
                if state.hasSelection {
                    Text("Selection: \(formatTime(state.selectionStart)) – \(formatTime(state.selectionEnd))")
                }
                Spacer()
                Text(formatTime(state.duration))
            }
            .font(.caption2)
            .foregroundStyle(.secondary)
            .padding(.horizontal, 16)
        }
    }

    private var frequencySection: some View {
        DisclosureGroup("Frequency Constraint") {
            VStack(spacing: 8) {
                Toggle("Enable frequency band", isOn: $state.frequencyBandEnabled)
                if state.frequencyBandEnabled {
                    HStack {
                        VStack(alignment: .leading) {
                            Text("Low Cut: \(Int(state.lowCutHz)) Hz").font(.caption)
                            Slider(value: $state.lowCutHz, in: 20...10000)
                        }
                        VStack(alignment: .leading) {
                            Text("High Cut: \(Int(state.highCutHz)) Hz").font(.caption)
                            Slider(value: $state.highCutHz, in: 100...20000)
                        }
                    }
                }
            }
            .padding(.vertical, 8)
        }
        .padding(.horizontal, 16)
    }

    private var emptyState: some View {
        VStack(spacing: 16) {
            Spacer()
            Image(systemName: "waveform.badge.plus")
                .font(.system(size: 48))
                .foregroundStyle(.secondary)
            Text("Open an audio file to start editing")
                .font(.headline).foregroundStyle(.secondary)
            Button("Open File") { openFile() }
                .buttonStyle(.bordered)
            Spacer()
        }
    }

    private func openFile() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.wav, .mp3, .audio]
        panel.allowsMultipleSelection = false
        panel.begin { response in
            guard response == .OK, let url = panel.url else { return }
            loadFromURL(url)
        }
    }

    private func loadFromURL(_ url: URL) {
        state.fileName = url.lastPathComponent

        do {
            let audioFile = try AVAudioFile(forReading: url)
            let format = audioFile.processingFormat
            let frameCount = AVAudioFrameCount(audioFile.length)

            guard let pcmBuffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
                return
            }
            try audioFile.read(into: pcmBuffer)

            let buffer = AudioBufferConverter.fromPCMBuffer(pcmBuffer)
            state.audioBuffer = buffer
            state.sampleRate = buffer.sampleRate
            state.duration = buffer.duration
            state.waveformPeaks = state.computePeaks(from: buffer)
            state.isLoaded = true
            state.selectionStart = 0
            state.selectionEnd = 0
            state.resetUndoHistory()

            // VAE-encode into the coordinator so inpainting has a starting latent.
            // Runs off the main thread — fires and forgets from the UI's perspective.
            Task {
                guard let coordinator = editorService.coordinator else { return }
                let mlxArray = AudioBufferConverter.toMLXArray(buffer)
                _ = try? await coordinator.loadAudio(
                    samples: mlxArray,
                    sampleRate: buffer.sampleRate,
                    precision: .float32
                )
            }
        } catch {
            state.isLoaded = false
        }
    }

    private func executeEdit() {
        guard let coordinator = editorService.coordinator else {
            state.progress = .failed("Editor pipeline not ready")
            return
        }
        guard !state.isProcessing else { return }

        let request = state.buildRequest()
        state.saveUndoSnapshot()   // save pre-inpaint buffer so AI edits are undoable
        state.isProcessing = true
        state.progress = .buildingMask

        Task { @MainActor in
            for await progress in await coordinator.edit(request: request) {
                state.progress = progress
                switch progress {
                case .complete, .failed:
                    state.isProcessing = false
                default:
                    break
                }
            }
            // Pull the regenerated audio back from the coordinator and update
            // the editor buffer so DSP tools and the waveform view stay in sync.
            if let updatedAudio = await coordinator.getCurrentAudio() {
                let newBuffer = AudioBufferConverter.fromMLXArray(
                    updatedAudio,
                    sampleRate: state.sampleRate
                )
                state.audioBuffer = newBuffer
                state.duration = newBuffer.duration
                state.waveformPeaks = state.computePeaks(from: newBuffer)
            }
        }
    }

    private func performUndo() { state.undoDSP() }
    private func performRedo() { state.redoDSP() }

    private func exportFile() {
        guard let buffer = state.audioBuffer else { return }

        let panel = NSSavePanel()
        panel.allowedContentTypes = [.wav]
        let baseName = (state.fileName as NSString).deletingPathExtension
        panel.nameFieldStringValue = baseName.isEmpty ? "edited" : "\(baseName)-edited"

        panel.begin { response in
            guard response == .OK, let url = panel.url else { return }
            do {
                let pcmBuffer = try AudioBufferConverter.toPCMBuffer(buffer)
                let outputFile = try AVAudioFile(forWriting: url, settings: pcmBuffer.format.settings)
                try outputFile.write(from: pcmBuffer)
            } catch {
                // Export failed silently — could surface an alert in a future pass
            }
        }
    }

    private func formatTime(_ time: TimeInterval) -> String {
        let minutes = Int(time) / 60
        let seconds = Int(time) % 60
        return String(format: "%d:%02d", minutes, seconds)
    }
}
