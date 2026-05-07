import SwiftUI
import AnvilCore

struct FirstLaunchView: View {
    @EnvironmentObject private var appState: AppState

    private var essentialSpecs: [ModelSpec] {
        ModelRegistry.builtInSpecs.filter {
            ModelRegistry.essentialModelIDs.contains($0.id)
        }
    }

    var body: some View {
        VStack(spacing: 32) {
            Spacer()

            Text("Welcome to Anvil")
                .font(.largeTitle.bold())

            Text("AI audio generation — music, sound effects, and more.\nCore models need to be downloaded before first use.")
                .multilineTextAlignment(.center)
                .foregroundStyle(.secondary)

            VStack(alignment: .leading, spacing: 8) {
                ForEach(essentialSpecs) { spec in
                    DownloadProgressRow(
                        spec: spec,
                        progress: appState.downloadProgress[spec.id]
                    )
                }
            }
            .frame(maxWidth: 400)
            .padding()
            .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12))

            if let error = appState.currentError {
                Label(error, systemImage: "exclamationmark.triangle.fill")
                    .foregroundStyle(.red)
            }

            totalSizeLabel

            Button(isActivelyDownloading ? "Downloading…" : "Download Models") {
                Task {
                    await appState.startDownload()
                }
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
            .disabled(isActivelyDownloading)

            Text("Additional models (sound effects, lyric writing) will download\nautomatically in the background after setup.")
                .font(.caption)
                .foregroundStyle(.tertiary)
                .multilineTextAlignment(.center)

            Spacer()
        }
        .padding(48)
        .frame(minWidth: 600, minHeight: 500)
    }

    private var isActivelyDownloading: Bool {
        !appState.downloadProgress.isEmpty &&
        appState.downloadProgress.values.contains(where: { $0 < 1.0 })
    }

    private var totalSizeLabel: some View {
        let totalBytes = essentialSpecs.reduce(Int64(0)) { $0 + $1.totalSizeBytes }
        let totalGB = Double(totalBytes) / 1_000_000_000
        return Text("Total download: \(String(format: "%.1f", totalGB)) GB — this only happens once.")
            .font(.caption)
            .foregroundStyle(.secondary)
    }
}
