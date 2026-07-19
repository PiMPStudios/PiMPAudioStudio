import AppKit
import SwiftUI

struct RunPodLifecyclePanel: View {
    @EnvironmentObject private var runPodService: RunPodService
    private static let runPodReferralURL = URL(string: "https://runpod.io?ref=sox5p475")!

    let hasSelectedPackage: Bool
    let preparedArchiveURL: URL?
    @Binding var apiKey: String
    @Binding var templateID: String
    @Binding var autoSelectTemplate: Bool
    @Binding var imageName: String
    @Binding var gpuType: String
    @Binding var interruptible: Bool
    @Binding var containerDiskGB: Int
    @Binding var volumeGB: Int
    @Binding var sshKeyPath: String
    let onLaunchRequested: () -> Void
    let onSelectArchiveRequested: () -> Void
    let onUploadRequested: (RunPodPodSummary) -> Void
    let onStartRequested: (RunPodPodSummary) -> Void
    let onMonitorRequested: (RunPodPodSummary) -> Void
    let onPullRequested: (RunPodPodSummary) -> Void
    let onStopRequested: (RunPodPodSummary) -> Void
    let onNotice: (String) -> Void

    @State private var showingAdvancedLaunchSource = false
    @State private var hasRequestedInitialLaunchOptions = false

    var body: some View {
        Group {
            runPodCredentialSection
            runPodLaunchSection
            runPodArchiveSection
            runPodPodsSection
        }
        .onChange(of: gpuType) { _, _ in
            applyRecommendedTemplateIfNeeded()
        }
        .onChange(of: autoSelectTemplate) { _, isAutomatic in
            if isAutomatic {
                applyRecommendedTemplateIfNeeded()
            }
        }
        .onChange(of: runPodService.podTemplates) { _, _ in
            applyRecommendedTemplateIfNeeded()
        }
        .task {
            await refreshInitialLaunchOptionsIfNeeded()
            applyRecommendedTemplateIfNeeded()
        }
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
                VStack(alignment: .leading, spacing: 3) {
                    Text("SSH Key")
                    Text(runPodSSHKeyStatus)
                        .font(.caption)
                        .foregroundStyle(hasConfiguredSSHKey ? .green : .secondary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }

                Spacer()

                Button {
                    chooseRunPodSSHKey()
                } label: {
                    Label(
                        hasConfiguredSSHKey ? "Replace\u{2026}" : "Choose Key\u{2026}",
                        systemImage: "key"
                    )
                }

                if runPodSSHKeyStore.hasManagedPrivateKey {
                    Button(role: .destructive) {
                        removeRunPodSSHKey()
                    } label: {
                        Image(systemName: "trash")
                    }
                    .buttonStyle(.borderless)
                    .help("Remove the RunPod SSH key from PiMP Audio Studio")
                }
            }

            Text("PiMP Audio Studio keeps a protected local copy in its sandbox for automated SSH and SCP. The private key is never added to the remote job zip.")
                .font(.caption2)
                .foregroundStyle(.secondary)

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
            runPodLaunchSourceSummary

            HStack {
                TextField("GPU Type", text: $gpuType)
                Menu(runPodService.gpuTypes.isEmpty ? "Common" : "GPU Types") {
                    if !runPodService.gpuTypes.isEmpty {
                        Section("RunPod") {
                            ForEach(runPodService.gpuTypes) { gpuType in
                                Button(gpuType.menuTitle) {
                                    selectGPUType(gpuType.selectionValue)
                                }
                            }
                        }

                        Divider()
                    }

                    Section(runPodService.gpuTypes.isEmpty ? "Common" : "Fallback") {
                        ForEach(commonRunPodGPUTypes, id: \.self) { gpuType in
                            Button(gpuType) {
                                selectGPUType(gpuType)
                            }
                        }
                    }
                }
                .menuStyle(.borderlessButton)

                Button {
                    Task { await runPodService.refreshLaunchOptions() }
                } label: {
                    Label("Refresh GPU + Templates", systemImage: "arrow.clockwise")
                }
                .disabled(!runPodService.hasAPIKey || runPodService.isRefreshingGPUTypes)
            }

            if runPodService.isRefreshingGPUTypes {
                HStack {
                    ProgressView()
                        .controlSize(.small)
                    Text("Loading GPU availability and official templates from RunPod\u{2026}")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }

            if let selectedRunPodGPUType {
                Text(selectedRunPodGPUType.detailLabel(interruptible: interruptible))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else if !runPodService.gpuTypes.isEmpty,
                      !gpuType.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                Text("Selected GPU type is not in the refreshed RunPod list. Manual values are still allowed.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            DisclosureGroup("Template Selection / Image", isExpanded: $showingAdvancedLaunchSource) {
                Toggle("Choose Best Template Automatically", isOn: $autoSelectTemplate)

                if autoSelectTemplate {
                    Text("PiMP selects the highest-ranked compatible official PyTorch template for the chosen GPU.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                } else {
                    Picker("Official Template", selection: manualTemplateSelection) {
                        Text("Custom Template ID").tag("")
                        ForEach(runPodService.podTemplates) { template in
                            Text(templatePickerLabel(template)).tag(template.id)
                        }
                    }

                    TextField("Template ID", text: $templateID)
                    TextField("Image", text: $imageName)

                    Text("Choose an official template above, enter any template ID, or provide a custom image. A custom image overrides the template ID.")
                        .font(.caption)
                        .foregroundStyle(.secondary)

                    if let selectedRunPodTemplate,
                       let reason = selectedRunPodTemplate.incompatibilityReason(for: gpuType) {
                        Label(reason, systemImage: "exclamationmark.triangle.fill")
                            .font(.caption)
                            .foregroundStyle(.orange)
                    }
                }
            }

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

            Text("Launch creates the RunPod pod. Once SSH is available, the pod row can upload, start, monitor, download, and import the reviewed job without manual Terminal steps.")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    private var runPodLaunchSourceSummary: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("Pod Template")
                    .font(.caption)
                    .fontWeight(.semibold)
                Spacer()
                if !autoSelectTemplate, recommendedRunPodTemplate != nil {
                    Button("Use Automatic") {
                        autoSelectTemplate = true
                        applyRecommendedTemplateIfNeeded()
                    }
                    .controlSize(.small)
                }
            }

            if autoSelectTemplate, let selectedRunPodTemplate {
                Label("Automatic: \(selectedRunPodTemplate.summaryLabel)", systemImage: "sparkles")
                    .font(.caption)
                    .foregroundStyle(.green)
                if !selectedRunPodTemplate.detailLabel.isEmpty {
                    Text(selectedRunPodTemplate.detailLabel)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            } else if autoSelectTemplate, runPodService.podTemplates.isEmpty {
                Text("Refresh GPU + Templates to choose a compatible official PyTorch template.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else if autoSelectTemplate {
                Label(
                    "No compatible official PyTorch template was found for \(gpuType).",
                    systemImage: "exclamationmark.triangle.fill"
                )
                .font(.caption)
                .foregroundStyle(.orange)
            } else if !imageName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                Label("Custom image \(imageName)", systemImage: "shippingbox")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else if let selectedRunPodTemplate {
                Label(selectedRunPodTemplate.summaryLabel, systemImage: "checkmark.circle.fill")
                    .font(.caption)
                    .foregroundStyle(.green)
                if !selectedRunPodTemplate.detailLabel.isEmpty {
                    Text(selectedRunPodTemplate.detailLabel)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            } else if !templateID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                Label("Custom template \(templateID)", systemImage: "slider.horizontal.3")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else if runPodService.hasAPIKey {
                Text("Choose an official template or enter a custom template ID.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else {
                Text("Save a RunPod API key to load official templates.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(10)
        .background(.secondary.opacity(0.06), in: RoundedRectangle(cornerRadius: 8))
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

    private var runPodArchiveSection: some View {
        Section("Remote Job Zip") {
            HStack(alignment: .firstTextBaseline, spacing: 12) {
                VStack(alignment: .leading, spacing: 3) {
                    Text(preparedArchiveURL?.lastPathComponent ?? "No job zip selected")
                        .lineLimit(1)
                        .truncationMode(.middle)

                    Text(preparedArchiveURL?.deletingLastPathComponent().path(percentEncoded: false) ?? "Create Remote Job fills this automatically. Choose an existing zip after restarting or launching from an older bundle.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                }

                Spacer()

                Button {
                    onSelectArchiveRequested()
                } label: {
                    Label(preparedArchiveURL == nil ? "Choose Job Zip\u{2026}" : "Change\u{2026}", systemImage: "doc.badge.plus")
                }
                .controlSize(.small)
            }

            Text("Upload and extract/run commands use this local zip. If those pod icons are disabled, select the exported remote job zip here first.")
                .font(.caption)
                .foregroundStyle(.secondary)
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

    private var runPodSSHKeyStore: RunPodSSHKeyStore {
        RunPodSSHKeyStore()
    }

    private var configuredSSHKeyURL: URL? {
        RunPodSSHKeyStore.configuredPrivateKeyURL(for: sshKeyPath)
    }

    private var hasConfiguredSSHKey: Bool {
        guard let configuredSSHKeyURL else { return false }
        var isDirectory: ObjCBool = false
        return FileManager.default.fileExists(
            atPath: configuredSSHKeyURL.path(percentEncoded: false),
            isDirectory: &isDirectory
        ) && !isDirectory.boolValue
    }

    private var runPodSSHKeyStatus: String {
        guard let configuredSSHKeyURL, hasConfiguredSSHKey else {
            return "Not configured"
        }
        return "Ready \u{00b7} \(configuredSSHKeyURL.path(percentEncoded: false))"
    }

    private var canLaunchRunPodPod: Bool {
        guard runPodService.hasAPIKey,
              hasSelectedPackage,
              !gpuType.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return false
        }
        if autoSelectTemplate {
            guard let recommendedRunPodTemplate else { return false }
            return templateID == recommendedRunPodTemplate.id
                && imageName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        }
        return !templateID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ||
            !imageName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    private var commonRunPodGPUTypes: [String] {
        [
            "NVIDIA RTX PRO 6000 Blackwell Server Edition",
            "NVIDIA RTX PRO 4500 Blackwell",
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

    private var selectedRunPodTemplate: RunPodTemplate? {
        runPodService.podTemplates.first { $0.matchesTemplateID(templateID) }
    }

    private var recommendedRunPodTemplate: RunPodTemplate? {
        runPodService.recommendedTrainingTemplate(for: gpuType)
    }

    private var manualTemplateSelection: Binding<String> {
        Binding(
            get: { selectedRunPodTemplate?.id ?? "" },
            set: { selectedID in
                guard let template = runPodService.podTemplates.first(where: {
                    $0.id == selectedID
                }) else {
                    templateID = ""
                    imageName = ""
                    return
                }
                applyTemplate(template)
            }
        )
    }

    private func selectGPUType(_ value: String) {
        gpuType = value
        applyRecommendedTemplateIfNeeded()
    }

    private func refreshInitialLaunchOptionsIfNeeded() async {
        guard !hasRequestedInitialLaunchOptions,
              runPodService.hasAPIKey,
              runPodService.gpuTypes.isEmpty || runPodService.podTemplates.isEmpty,
              !runPodService.isRefreshingGPUTypes else {
            return
        }
        hasRequestedInitialLaunchOptions = true
        await runPodService.refreshLaunchOptions()
    }

    private func applyRecommendedTemplateIfNeeded() {
        guard autoSelectTemplate else { return }
        guard let recommendedRunPodTemplate else {
            templateID = ""
            imageName = ""
            return
        }
        applyTemplate(recommendedRunPodTemplate)
    }

    private func applyTemplate(_ template: RunPodTemplate) {
        templateID = template.id
        imageName = ""

        if let templateContainerDiskGB = template.containerDiskInGb,
           templateContainerDiskGB > containerDiskGB {
            containerDiskGB = templateContainerDiskGB
        }
        if let templateVolumeGB = template.volumeInGb,
           templateVolumeGB > volumeGB {
            volumeGB = templateVolumeGB
        }
    }

    private func templatePickerLabel(_ template: RunPodTemplate) -> String {
        let compatibilitySuffix = template.incompatibilityReason(for: gpuType) == nil
            ? ""
            : " — incompatible"
        if let imageName = template.imageNameValue {
            return "\(template.displayNameValue) · \(imageName)\(compatibilitySuffix)"
        }
        return "\(template.displayNameValue)\(compatibilitySuffix)"
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

                if runPodService.terminatingPodID == pod.id {
                    ProgressView()
                        .controlSize(.small)
                }

                Button {
                    copyRunPodUploadCommand(for: pod)
                } label: {
                    Image(systemName: "shippingbox")
                }
                .buttonStyle(.borderless)
                .disabled(pod.uploadCommand(archiveURL: preparedArchiveURL, sshKeyPath: sshKeyPath) == nil)
                .help("Copy upload command for the job zip and PiMP worker")

                Button {
                    copyRunPodSSHCommand(for: pod)
                } label: {
                    Image(systemName: "terminal")
                }
                .buttonStyle(.borderless)
                .disabled(pod.sshCommand(sshKeyPath: sshKeyPath) == nil)
                .help("Copy SSH command")

                Button {
                    copyRunPodRemotePackageCommand(for: pod)
                } label: {
                    Image(systemName: "play.rectangle")
                }
                .buttonStyle(.borderless)
                .disabled(pod.remotePackageCommand(archiveURL: preparedArchiveURL) == nil)
                .help("Copy extract and run command for the pod")

                Button(role: .destructive) {
                    onStopRequested(pod)
                } label: {
                    Image(systemName: "trash")
                }
                .buttonStyle(.borderless)
                .disabled(runPodService.terminatingPodID != nil || pod.locked == true)
                .help("Terminate this RunPod pod")
            }

            if let publicIp = pod.publicIp, let sshPort = pod.sshPort {
                Text(verbatim: "SSH \(publicIp):\(sshPort)")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
            }

            if preparedArchiveURL != nil {
                automatedRemoteJobControls(for: pod)
            }
        }
        .padding(.vertical, 4)
    }

    private func automatedRemoteJobControls(
        for pod: RunPodPodSummary
    ) -> some View {
        let state = runPodService.remoteJobState(
            for: pod,
            archiveURL: preparedArchiveURL
        )
        return VStack(alignment: .leading, spacing: 5) {
            HStack(spacing: 6) {
                Button("Upload") {
                    onUploadRequested(pod)
                }
                .disabled(
                    preparedArchiveURL == nil
                        || !hasConfiguredSSHKey
                        || (state != nil && state?.phase != .failed)
                        || pod.sshPort == nil
                )

                Button("Start") {
                    onStartRequested(pod)
                }
                .disabled(state?.phase != .uploaded)

                Button("Monitor") {
                    onMonitorRequested(pod)
                }
                .disabled(
                    !RunPodRemoteJobControlAvailability.canMonitor(
                        phase: state?.phase,
                        hasArchive: preparedArchiveURL != nil,
                        hasSSHKey: hasConfiguredSSHKey,
                        hasSSHEndpoint: pod.sshPort != nil
                    )
                )
                .help(
                    state == nil
                        ? "Reconnect to the remote job and restore its status"
                        : "Refresh the remote job status"
                )

                Button("Pull & Import") {
                    onPullRequested(pod)
                }
                .buttonStyle(.borderedProminent)
                .disabled(state?.phase != .succeeded)

                Spacer()

                if state?.phase.isBusy == true {
                    ProgressView()
                        .controlSize(.small)
                }
            }
            .controlSize(.small)

            if let state {
                Label(
                    state.message,
                    systemImage: remoteJobStatusIcon(state.phase)
                )
                .font(.caption2)
                .foregroundStyle(remoteJobStatusColor(state.phase))
                .lineLimit(3)
                .textSelection(.enabled)

                if !state.logTail.isEmpty {
                    Text(state.logTail)
                        .font(.system(.caption2, design: .monospaced))
                        .foregroundStyle(.secondary)
                        .lineLimit(5)
                        .textSelection(.enabled)
                }
            } else {
                Text("No local status is loaded. Monitor reconnects to the selected job already on this pod; Upload replaces it.")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        }
    }

    private func remoteJobStatusIcon(
        _ phase: RunPodRemoteJobPhase
    ) -> String {
        switch phase {
        case .uploaded, .succeeded, .downloaded, .imported:
            return "checkmark.circle.fill"
        case .failed:
            return "exclamationmark.triangle.fill"
        case .uploading, .starting, .running, .downloading:
            return "clock.arrow.circlepath"
        }
    }

    private func remoteJobStatusColor(
        _ phase: RunPodRemoteJobPhase
    ) -> Color {
        switch phase {
        case .uploaded, .succeeded, .downloaded, .imported:
            return .green
        case .failed:
            return .orange
        case .uploading, .starting, .running, .downloading:
            return .secondary
        }
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
        hasRequestedInitialLaunchOptions = true
        Task {
            await runPodService.refreshLaunchOptions()
        }
    }

    private func chooseRunPodSSHKey() {
        let panel = NSOpenPanel()
        panel.title = "Choose RunPod SSH Private Key"
        panel.message = "Choose the private key whose public key is registered with RunPod."
        panel.prompt = "Use SSH Key"
        panel.canChooseDirectories = false
        panel.canChooseFiles = true
        panel.allowsMultipleSelection = false
        panel.directoryURL = RunPodSSHKeyStore.suggestedSourceDirectoryURL

        guard panel.runModal() == .OK, let selectedURL = panel.url else {
            return
        }

        do {
            let managedURL = try runPodSSHKeyStore.importPrivateKey(
                from: selectedURL
            )
            sshKeyPath = managedURL.path(percentEncoded: false)
            onNotice("RunPod SSH key is ready for automated uploads.")
        } catch {
            onNotice(error.localizedDescription)
        }
    }

    private func removeRunPodSSHKey() {
        do {
            try runPodSSHKeyStore.removeManagedPrivateKey()
            sshKeyPath = ""
            onNotice("Removed the RunPod SSH key from PiMP Audio Studio.")
        } catch {
            onNotice("Could not remove the RunPod SSH key: \(error.localizedDescription)")
        }
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
        onNotice("Copied upload command for \(pod.displayName). It uploads the job zip and PiMP worker.")
    }

    private func copyRunPodRemotePackageCommand(for pod: RunPodPodSummary) {
        guard let command = pod.remotePackageCommand(archiveURL: preparedArchiveURL) else {
            onNotice("Export a remote job zip first, then copy the pod run command.")
            return
        }
        copyToPasteboard(command)
        onNotice("Copied extract/run command for \(pod.displayName).")
    }

    private func copyToPasteboard(_ value: String) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(value, forType: .string)
    }
}
