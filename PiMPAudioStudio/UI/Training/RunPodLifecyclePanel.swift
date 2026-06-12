import AppKit
import SwiftUI

struct RunPodLifecyclePanel: View {
    @EnvironmentObject private var runPodService: RunPodService
    private static let runPodReferralURL = URL(string: "https://runpod.io?ref=sox5p475")!

    let hasSelectedPackage: Bool
    let preparedArchiveURL: URL?
    @Binding var apiKey: String
    @Binding var templateID: String
    @Binding var imageName: String
    @Binding var gpuType: String
    @Binding var cloudTypeRaw: String
    @Binding var interruptible: Bool
    @Binding var containerDiskGB: Int
    @Binding var volumeGB: Int
    @Binding var sshKeyPath: String
    let onLaunchRequested: () -> Void
    let onStopRequested: (RunPodPodSummary) -> Void
    let onNotice: (String) -> Void

    var body: some View {
        runPodCredentialSection
        runPodLaunchSection
        runPodPodsSection
    }

    private var runPodCredentialSection: some View {
        Section("RunPod") {
            HStack {
                VStack(alignment: .leading, spacing: 3) {
                    Text("API Key")
                    Text(runPodCredentialStatus)
                        .font(.caption)
                        .foregroundStyle(runPodService.hasAPIKey ? .green : .secondary)
                }

                Spacer()

                SecureField("RunPod API Key", text: $apiKey)
                    .frame(width: 180)
                    .onSubmit(saveRunPodAPIKey)

                Button("Save") {
                    saveRunPodAPIKey()
                }
                .disabled(apiKey.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)

                if runPodService.keySource == .keychain {
                    Button(role: .destructive) {
                        runPodService.removeAPIKey()
                    } label: {
                        Image(systemName: "trash")
                    }
                    .buttonStyle(.borderless)
                    .help("Remove RunPod API key")
                }
            }

            HStack {
                Button {
                    Task { await runPodService.refreshPods() }
                } label: {
                    Label("Refresh Pods", systemImage: "arrow.clockwise")
                }
                .disabled(!runPodService.hasAPIKey || runPodService.isRefreshing)

                Button {
                    NSWorkspace.shared.open(URL(string: "https://console.runpod.io/pods")!)
                } label: {
                    Label("Console", systemImage: "safari")
                }

                Button {
                    NSWorkspace.shared.open(Self.runPodReferralURL)
                } label: {
                    Label("Create Account", systemImage: "person.crop.circle.badge.plus")
                }

                Spacer()

                if runPodService.isRefreshing {
                    ProgressView()
                        .controlSize(.small)
                }
            }

            if let message = runPodService.lastMessage {
                Text(message)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            if let error = runPodService.lastError {
                Label(error, systemImage: "exclamationmark.triangle.fill")
                    .font(.caption)
                    .foregroundStyle(.orange)
            }

            if !runPodService.hasAPIKey {
                Text("No RunPod account yet? Create one, generate an API key in RunPod, then paste it here. You can still export the remote job zip without connecting RunPod.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }

    private var runPodLaunchSection: some View {
        Section("Launch Pod") {
            TextField("Template ID", text: $templateID)
            TextField("Image", text: $imageName)

            HStack {
                TextField("GPU Type", text: $gpuType)
                Menu(runPodService.gpuTypes.isEmpty ? "Common" : "GPU Types") {
                    if !runPodService.gpuTypes.isEmpty {
                        Section("RunPod") {
                            ForEach(runPodService.gpuTypes) { gpuType in
                                Button(gpuType.menuTitle) {
                                    self.gpuType = gpuType.selectionValue
                                }
                            }
                        }

                        Divider()
                    }

                    Section(runPodService.gpuTypes.isEmpty ? "Common" : "Fallback") {
                        ForEach(commonRunPodGPUTypes, id: \.self) { gpuType in
                            Button(gpuType) {
                                self.gpuType = gpuType
                            }
                        }
                    }
                }
                .menuStyle(.borderlessButton)

                Button {
                    Task { await runPodService.refreshGPUTypes() }
                } label: {
                    Label("Refresh GPU Types", systemImage: "arrow.clockwise")
                }
                .disabled(!runPodService.hasAPIKey || runPodService.isRefreshingGPUTypes)
            }

            if runPodService.isRefreshingGPUTypes {
                HStack {
                    ProgressView()
                        .controlSize(.small)
                    Text("Loading GPU availability from RunPod\u{2026}")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }

            if let selectedRunPodGPUType {
                Text(selectedRunPodGPUType.detailLabel)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else if !runPodService.gpuTypes.isEmpty,
                      !gpuType.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                Text("Selected GPU type is not in the refreshed RunPod list. Manual values are still allowed.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Picker("Cloud", selection: runPodCloudTypeBinding) {
                ForEach(RunPodCloudType.allCases) { cloudType in
                    Text(cloudType.title).tag(cloudType)
                }
            }
            .pickerStyle(.segmented)

            Toggle("Interruptible", isOn: $interruptible)

            Stepper(value: $containerDiskGB, in: 20...500, step: 10) {
                LabeledContent("Container Disk", value: "\(containerDiskGB) GB")
            }

            Stepper(value: $volumeGB, in: 20...1_000, step: 10) {
                LabeledContent("Volume", value: "\(volumeGB) GB")
            }

            Button {
                onLaunchRequested()
            } label: {
                Label("Launch RunPod Pod\u{2026}", systemImage: "play.circle")
            }
            .buttonStyle(.borderedProminent)
            .disabled(!canLaunchRunPodPod || runPodService.isLaunching)

            if runPodService.isLaunching {
                HStack {
                    ProgressView()
                        .controlSize(.small)
                    Text("Requesting pod launch\u{2026}")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }

            Text("Launch only creates the RunPod pod. Upload the exported job zip to the pod, run the included script, then pull artifacts back for import.")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    private var runPodPodsSection: some View {
        Section("Pods") {
            if runPodService.pods.isEmpty {
                Text(runPodService.hasAPIKey ? "No pods loaded. Refresh to list RunPod pods." : "Add a RunPod API key to list pods.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else {
                ForEach(runPodService.pods) { pod in
                    runPodPodRow(pod)
                }
            }
        }
    }

    private var runPodCredentialStatus: String {
        switch runPodService.keySource {
        case .keychain:
            return "Configured in Keychain"
        case .environment:
            return "From environment (\(APIKeyStore.runPodProvider.envVar))"
        case .none:
            return "Not configured"
        }
    }

    private var runPodCloudTypeBinding: Binding<RunPodCloudType> {
        Binding(
            get: { RunPodCloudType(rawValue: cloudTypeRaw) ?? .secure },
            set: { cloudTypeRaw = $0.rawValue }
        )
    }

    private var canLaunchRunPodPod: Bool {
        guard runPodService.hasAPIKey,
              hasSelectedPackage,
              !gpuType.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return false
        }
        return !templateID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ||
            !imageName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    private var commonRunPodGPUTypes: [String] {
        [
            "NVIDIA RTX PRO 6000 Blackwell Server Edition",
            "NVIDIA RTX 6000 Ada Generation",
            "NVIDIA GeForce RTX 5090",
            "NVIDIA GeForce RTX 4090",
            "NVIDIA L40S",
            "NVIDIA A100-SXM4-80GB",
            "NVIDIA H100 80GB HBM3",
        ]
    }

    private var selectedRunPodGPUType: RunPodGPUType? {
        runPodService.gpuTypes.first { $0.matches(gpuType) }
    }

    private func runPodPodRow(_ pod: RunPodPodSummary) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .firstTextBaseline, spacing: 8) {
                Image(systemName: runPodStatusIcon(for: pod))
                    .foregroundStyle(runPodStatusColor(for: pod))
                    .frame(width: 18)

                VStack(alignment: .leading, spacing: 2) {
                    Text(pod.displayName)
                        .font(.callout)
                        .lineLimit(1)
                        .truncationMode(.middle)

                    HStack(spacing: 8) {
                        Text(pod.statusLabel)
                        Text(pod.gpuLabel)
                        if let hourlyCostLabel = pod.hourlyCostLabel {
                            Text(hourlyCostLabel)
                        }
                    }
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                }

                Spacer()

                if runPodService.stoppingPodID == pod.id {
                    ProgressView()
                        .controlSize(.small)
                }

                Button {
                    copyRunPodSSHCommand(for: pod)
                } label: {
                    Image(systemName: "terminal")
                }
                .buttonStyle(.borderless)
                .disabled(pod.sshCommand(sshKeyPath: sshKeyPath) == nil)
                .help("Copy SSH command")

                Button {
                    copyRunPodUploadCommand(for: pod)
                } label: {
                    Image(systemName: "shippingbox")
                }
                .buttonStyle(.borderless)
                .disabled(pod.uploadCommand(archiveURL: preparedArchiveURL, sshKeyPath: sshKeyPath) == nil)
                .help("Copy upload command for the last exported job zip")

                Button(role: .destructive) {
                    onStopRequested(pod)
                } label: {
                    Image(systemName: "stop.circle")
                }
                .buttonStyle(.borderless)
                .disabled(runPodService.stoppingPodID != nil || pod.locked == true)
                .help("Stop this RunPod pod")
            }

            if let publicIp = pod.publicIp, let sshPort = pod.sshPort {
                Text("SSH \(publicIp):\(sshPort)")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
            }
        }
        .padding(.vertical, 4)
    }

    private func runPodStatusIcon(for pod: RunPodPodSummary) -> String {
        switch pod.statusLabel.uppercased() {
        case "RUNNING":
            return "checkmark.circle.fill"
        case "EXITED", "TERMINATED":
            return "pause.circle.fill"
        default:
            return "circle.dashed"
        }
    }

    private func runPodStatusColor(for pod: RunPodPodSummary) -> Color {
        switch pod.statusLabel.uppercased() {
        case "RUNNING":
            return .green
        case "EXITED", "TERMINATED":
            return .secondary
        default:
            return .orange
        }
    }

    private func saveRunPodAPIKey() {
        runPodService.saveAPIKey(apiKey)
        apiKey = ""
    }

    private func copyRunPodSSHCommand(for pod: RunPodPodSummary) {
        guard let command = pod.sshCommand(sshKeyPath: sshKeyPath) else { return }
        copyToPasteboard(command)
        onNotice("Copied SSH command for \(pod.displayName).")
    }

    private func copyRunPodUploadCommand(for pod: RunPodPodSummary) {
        guard let command = pod.uploadCommand(archiveURL: preparedArchiveURL, sshKeyPath: sshKeyPath) else {
            onNotice("Export a remote job zip first, then refresh pods and copy the upload command.")
            return
        }
        copyToPasteboard(command)
        onNotice("Copied upload command for \(pod.displayName).")
    }

    private func copyToPasteboard(_ value: String) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(value, forType: .string)
    }
}
