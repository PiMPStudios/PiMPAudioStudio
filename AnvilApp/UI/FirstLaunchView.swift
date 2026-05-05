import SwiftUI
import AnvilCore

struct FirstLaunchView: View {
    @EnvironmentObject private var appState: AppState
    @State private var isDownloading = false

    var body: some View {
        VStack(spacing: 32) {
            Spacer()

            Text("Welcome to Anvil")
                .font(.largeTitle.bold())

            Text("AI audio generation — music, sound effects, and more.\nModels need to be downloaded before first use.")
                .multilineTextAlignment(.center)
                .foregroundStyle(.secondary)

            VStack(alignment: .leading, spacing: 8) {
                ForEach(appState.registry.allSpecs) { spec in
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

            Button(isDownloading ? "Downloading…" : "Download Models") {
                isDownloading = true
                Task {
                    await appState.startDownload()
                    isDownloading = false
                }
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
            .disabled(isDownloading)

            Spacer()
        }
        .padding(48)
        .frame(minWidth: 600, minHeight: 500)
    }

    private var totalSizeLabel: some View {
        let totalBytes = appState.registry.allSpecs.reduce(Int64(0)) { $0 + $1.sizeBytes }
        let totalGB = Double(totalBytes) / 1_000_000_000
        return Text("Total download: \(String(format: "%.1f", totalGB)) GB — this only happens once.")
            .font(.caption)
            .foregroundStyle(.secondary)
    }
}
