import SwiftUI
import UniformTypeIdentifiers

struct PlayerView: View {
    @EnvironmentObject private var playerService: AudioPlayerService

    let fileURL: URL
    /// Pre-computed waveform peaks from generation time.
    /// When provided the player skips re-reading the WAV file for peak extraction.
    var precomputedPeaks: [Float]? = nil

    var body: some View {
        VStack(spacing: 12) {
            HStack(spacing: 12) {
                Button(action: { playerService.togglePlayback() }) {
                    Image(systemName: playerService.isPlaying ? "pause.circle.fill" : "play.circle.fill")
                        .font(.system(size: 32))
                        .foregroundStyle(.tint)
                }
                .buttonStyle(.plain)

                if let waveform = playerService.waveform {
                    WaveformView(
                        peaks: waveform.peaks,
                        progress: playerService.duration > 0
                            ? playerService.currentTime / playerService.duration
                            : 0,
                        onSeek: { fraction in
                            let time = fraction * playerService.duration
                            playerService.seek(to: time)
                        }
                    )
                    .frame(height: 40)
                } else {
                    ProgressView()
                        .frame(height: 40)
                }

                Text(timeString(playerService.currentTime) + " / " + timeString(playerService.duration))
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .monospacedDigit()
                    .frame(minWidth: 70)
            }

            HStack(spacing: 12) {
                Image(systemName: "speaker.fill")
                    .font(.caption)
                    .foregroundStyle(.secondary)

                Slider(value: Binding(
                    get: { playerService.volume },
                    set: { playerService.setVolume($0) }
                ), in: 0...1)
                .frame(width: 80)

                Spacer()

                // Share sheet — AirDrop, Mail, Messages, Notes, etc.
                ShareLink(
                    item: fileURL,
                    preview: SharePreview(
                        fileURL.deletingPathExtension().lastPathComponent,
                        icon: Image(systemName: "waveform")
                    )
                ) {
                    Label("Share", systemImage: "square.and.arrow.up")
                }
                .buttonStyle(.bordered)
                .controlSize(.small)

                Button("Export WAV") {
                    exportToFinder()
                }
                .buttonStyle(.bordered)
                .controlSize(.small)

                Button("Reveal in Finder") {
                    NSWorkspace.shared.activateFileViewerSelecting([fileURL])
                }
                .buttonStyle(.bordered)
                .controlSize(.small)

                Button("Edit This") {
                    NotificationCenter.default.post(
                        name: .navigateToEditor,
                        object: nil,
                        userInfo: ["url": fileURL]
                    )
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            }
        }
        .padding()
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
        .onAppear {
            playerService.load(url: fileURL, precomputedPeaks: precomputedPeaks)
        }
        .onChange(of: fileURL) { _, newURL in
            playerService.load(url: newURL, precomputedPeaks: precomputedPeaks)
        }
    }

    private func timeString(_ time: TimeInterval) -> String {
        let minutes = Int(time) / 60
        let seconds = Int(time) % 60
        return String(format: "%d:%02d", minutes, seconds)
    }

    private func exportToFinder() {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.wav]
        panel.nameFieldStringValue = fileURL.lastPathComponent
        panel.begin { response in
            guard response == .OK, let destination = panel.url else { return }
            try? FileManager.default.copyItem(at: fileURL, to: destination)
        }
    }
}

extension Notification.Name {
    static let navigateToEditor = Notification.Name("navigateToEditor")
}
