import SwiftUI
import AnvilCore

struct EditorView: View {
    @StateObject private var state = EditorState()

    var preloadedAudioURL: URL?
    var preloadedPrompt: String?

    var body: some View {
        VStack(spacing: 0) {
            EditorToolbar(
                state: state,
                onOpen: openFile,
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
        state.isLoaded = true
        state.duration = 30.0
        state.waveformPeaks = (0..<200).map { _ in Float.random(in: 0.1...0.9) }
    }

    private func executeEdit() {
        state.isProcessing = true
        state.progress = .buildingMask
    }

    private func performUndo() {}
    private func performRedo() {}

    private func formatTime(_ time: TimeInterval) -> String {
        let minutes = Int(time) / 60
        let seconds = Int(time) % 60
        return String(format: "%d:%02d", minutes, seconds)
    }
}
