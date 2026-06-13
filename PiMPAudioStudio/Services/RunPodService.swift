import Foundation
import PiMPAudioCore
import SwiftUI

@MainActor
final class RunPodService: ObservableObject {
    @Published private(set) var pods: [RunPodPodSummary] = []
    @Published private(set) var gpuTypes: [RunPodGPUType] = []
    @Published private(set) var podTemplates: [RunPodTemplate] = []
    @Published private(set) var isRefreshing = false
    @Published private(set) var isRefreshingGPUTypes = false
    @Published private(set) var isLaunching = false
    @Published private(set) var terminatingPodID: String?
    @Published var lastMessage: String?
    @Published var lastError: String?

    private let apiKeyStore: APIKeyStore
    private let client: RunPodAPIClient

    init(
        apiKeyStore: APIKeyStore,
        client: RunPodAPIClient = RunPodAPIClient()
    ) {
        self.apiKeyStore = apiKeyStore
        self.client = client
    }

    var hasAPIKey: Bool {
        apiKeyStore.hasKey(for: APIKeyStore.runPodProviderID)
    }

    var keySource: APIKeyStore.KeySource {
        apiKeyStore.keySource(for: APIKeyStore.runPodProviderID)
    }

    func saveAPIKey(_ key: String) {
        let trimmed = key.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        apiKeyStore.setAPIKey(trimmed, for: APIKeyStore.runPodProviderID)
        note("apiKeySaved", "source=\(keySourceDiagnosticsLabel)")
        lastMessage = "RunPod API key saved."
        lastError = nil
    }

    func removeAPIKey() {
        apiKeyStore.removeAPIKey(for: APIKeyStore.runPodProviderID)
        pods = []
        gpuTypes = []
        podTemplates = []
        note("apiKeyRemoved", "source=none")
        lastMessage = "RunPod API key removed."
        lastError = nil
    }

    func refreshPods() async {
        guard let token = runPodToken() else {
            lastError = "Add a RunPod API key before refreshing pods."
            return
        }

        isRefreshing = true
        defer { isRefreshing = false }

        do {
            note("pods/refreshBegin", "source=\(keySourceDiagnosticsLabel)")
            pods = try await client.listPods(token: token)
            note(
                "pods/refreshEnd",
                "count=\(pods.count) ids=\(pods.map(\.id).prefix(12).joined(separator: ","))"
            )
            lastMessage = "Loaded \(pods.count) RunPod pod\(pods.count == 1 ? "" : "s")."
            lastError = nil
        } catch {
            note("pods/refreshError", "error=\(RunPodError.userFacingMessage(for: error))")
            lastError = RunPodError.userFacingMessage(for: error)
        }
    }

    func refreshGPUTypes() async {
        guard let token = runPodToken() else {
            lastError = "Add a RunPod API key before refreshing GPU types."
            return
        }

        isRefreshingGPUTypes = true
        defer { isRefreshingGPUTypes = false }

        do {
            note("gpuTypes/refreshBegin", "source=\(keySourceDiagnosticsLabel)")
            gpuTypes = try await client.listGPUTypes(token: token)
            note(
                "gpuTypes/refreshEnd",
                "count=\(gpuTypes.count) preview=\(gpuTypes.map(\.selectionValue).prefix(8).joined(separator: " | "))"
            )
            lastMessage = "Loaded \(gpuTypes.count) RunPod GPU type\(gpuTypes.count == 1 ? "" : "s")."
            lastError = nil
        } catch {
            note("gpuTypes/refreshError", "error=\(RunPodError.userFacingMessage(for: error))")
            lastError = RunPodError.userFacingMessage(for: error)
        }
    }

    func refreshLaunchOptions() async {
        guard let token = runPodToken() else {
            lastError = "Add a RunPod API key before refreshing RunPod launch options."
            return
        }

        isRefreshingGPUTypes = true
        defer { isRefreshingGPUTypes = false }

        do {
            note("launchOptions/refreshBegin", "source=\(keySourceDiagnosticsLabel)")
            let loadedGPUTypes = try await client.listGPUTypes(token: token)
            let loadedTemplates = try await client.listOfficialPodTemplates(token: token)
            gpuTypes = loadedGPUTypes
            podTemplates = loadedTemplates
            let recommended = RunPodTemplate.recommendedTrainingTemplate(
                for: loadedGPUTypes.first?.selectionValue ?? "",
                from: loadedTemplates
            )
            note(
                "launchOptions/refreshEnd",
                "gpuCount=\(loadedGPUTypes.count) templateCount=\(loadedTemplates.count) firstGPU=\(loadedGPUTypes.first?.selectionValue ?? "none") recommendedTemplate=\(recommended?.id ?? "none")"
            )
            lastMessage = "Loaded \(loadedGPUTypes.count) GPU type\(loadedGPUTypes.count == 1 ? "" : "s") and \(loadedTemplates.count) official RunPod template\(loadedTemplates.count == 1 ? "" : "s")."
            lastError = nil
        } catch {
            note("launchOptions/refreshError", "error=\(RunPodError.userFacingMessage(for: error))")
            lastError = RunPodError.userFacingMessage(for: error)
        }
    }

    func recommendedTrainingTemplate(for gpuTypeValue: String) -> RunPodTemplate? {
        RunPodTemplate.recommendedTrainingTemplate(
            for: gpuTypeValue,
            from: podTemplates
        )
    }

    @discardableResult
    func launchPod(configuration: RunPodLaunchConfiguration) async -> RunPodPodSummary? {
        guard let token = runPodToken() else {
            lastError = "Add a RunPod API key before launching a pod."
            return nil
        }

        isLaunching = true
        defer { isLaunching = false }

        do {
            note("launch/begin", configuration.diagnosticsSummary)
            let pod = try await client.createPod(token: token, configuration: configuration)
            upsert(pod)
            note("launch/success", pod.diagnosticsSummary)
            lastMessage = "RunPod launch requested for \(pod.displayName)."
            lastError = nil
            return pod
        } catch {
            note("launch/error", "configuration={\(configuration.diagnosticsSummary)} error=\(RunPodError.userFacingMessage(for: error))")
            lastError = RunPodError.userFacingMessage(for: error)
            return nil
        }
    }

    func terminatePod(id: String) async {
        guard let token = runPodToken() else {
            lastError = "Add a RunPod API key before terminating a pod."
            return
        }

        terminatingPodID = id
        defer { terminatingPodID = nil }

        do {
            note("terminate/begin", "podID=\(id)")
            try await client.deletePod(token: token, podID: id)
            pods.removeAll { $0.id == id }
            note("terminate/success", "podID=\(id)")
            lastMessage = "Terminate requested for RunPod pod \(id)."
            lastError = nil
            await refreshPods()
        } catch {
            note("terminate/error", "podID=\(id) error=\(RunPodError.userFacingMessage(for: error))")
            lastError = RunPodError.userFacingMessage(for: error)
        }
    }

    var stoppingPodID: String? {
        terminatingPodID
    }

    private func runPodToken() -> String? {
        apiKeyStore.runPodToken()
    }

    private func upsert(_ pod: RunPodPodSummary) {
        if let index = pods.firstIndex(where: { $0.id == pod.id }) {
            pods[index] = pod
        } else {
            pods.insert(pod, at: 0)
        }
    }

    private var keySourceDiagnosticsLabel: String {
        switch keySource {
        case .keychain:
            return "keychain"
        case .environment:
            return "environment"
        case .none:
            return "none"
        }
    }

    private func note(_ tag: String, _ message: String) {
        AudioDiagnostics.note("runpod/\(tag)", message)
    }
}

struct RunPodLaunchConfiguration: Equatable, Sendable {
    var name: String
    var templateID: String
    var imageName: String
    var gpuTypeID: String
    var interruptible: Bool
    var containerDiskGB: Int
    var volumeGB: Int
    var ports: [String] = ["22/tcp", "8888/http"]
    var supportPublicIP: Bool = true

    func requestBody() throws -> RunPodCreatePodRequest {
        let trimmedName = name.trimmingCharacters(in: .whitespacesAndNewlines)
        let trimmedTemplateID = templateID.trimmingCharacters(in: .whitespacesAndNewlines)
        let trimmedImageName = imageName.trimmingCharacters(in: .whitespacesAndNewlines)
        let trimmedGPUType = gpuTypeID.trimmingCharacters(in: .whitespacesAndNewlines)

        guard !trimmedTemplateID.isEmpty || !trimmedImageName.isEmpty else {
            throw RunPodError.invalidConfiguration("Enter a RunPod template ID or container image.")
        }
        guard !trimmedGPUType.isEmpty else {
            throw RunPodError.invalidConfiguration("Enter at least one RunPod GPU type.")
        }
        guard containerDiskGB >= 20 else {
            throw RunPodError.invalidConfiguration("Container disk must be at least 20 GB.")
        }
        guard volumeGB >= 20 else {
            throw RunPodError.invalidConfiguration("Volume must be at least 20 GB.")
        }

        let requestTemplateID = trimmedImageName.isEmpty ? trimmedTemplateID : ""
        let requestImageName = trimmedImageName

        return RunPodCreatePodRequest(
            cloudType: "SECURE",
            computeType: "GPU",
            containerDiskInGb: containerDiskGB,
            gpuCount: 1,
            gpuTypeIds: [trimmedGPUType],
            gpuTypePriority: "availability",
            imageName: requestImageName.isEmpty ? nil : requestImageName,
            interruptible: interruptible,
            locked: false,
            name: trimmedName.isEmpty ? "PiMP Audio Studio Remote Job" : trimmedName,
            ports: ports,
            supportPublicIp: supportPublicIP,
            templateId: requestTemplateID.isEmpty ? nil : requestTemplateID,
            volumeInGb: volumeGB,
            volumeMountPath: "/workspace"
        )
    }

    var diagnosticsSummary: String {
        [
            "name=\(name)",
            "templateID=\(templateID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? "none" : templateID)",
            "image=\(imageName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? "none" : imageName)",
            "gpu=\(gpuTypeID)",
            "interruptible=\(interruptible)",
            "containerDiskGB=\(containerDiskGB)",
            "volumeGB=\(volumeGB)"
        ].joined(separator: " ")
    }
}

struct RunPodPodSummary: Identifiable, Decodable, Equatable, Sendable {
    var id: String
    var name: String?
    var desiredStatus: String?
    var image: String?
    var interruptible: Bool?
    var locked: Bool?
    var costPerHr: RunPodFlexibleDouble?
    var adjustedCostPerHr: RunPodFlexibleDouble?
    var gpu: RunPodGPU?
    var machine: RunPodMachine?
    var gpuCount: Int?
    var memoryInGb: Double?
    var publicIp: String?
    var portMappings: [String: Int]
    var ports: [String]
    var templateId: String?
    var lastStartedAt: String?
    var lastStatusChange: String?
    var containerDiskInGb: Int?
    var volumeInGb: Int?
    var volumeMountPath: String?

    enum CodingKeys: String, CodingKey {
        case id
        case name
        case desiredStatus
        case image
        case interruptible
        case locked
        case costPerHr
        case adjustedCostPerHr
        case gpu
        case machine
        case gpuCount
        case memoryInGb
        case publicIp
        case portMappings
        case ports
        case templateId
        case lastStartedAt
        case lastStatusChange
        case containerDiskInGb
        case volumeInGb
        case volumeMountPath
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(String.self, forKey: .id)
        name = try container.decodeIfPresent(String.self, forKey: .name)
        desiredStatus = try container.decodeIfPresent(String.self, forKey: .desiredStatus)
        image = try container.decodeIfPresent(String.self, forKey: .image)
        interruptible = try container.decodeIfPresent(Bool.self, forKey: .interruptible)
        locked = try container.decodeIfPresent(Bool.self, forKey: .locked)
        costPerHr = try container.decodeIfPresent(RunPodFlexibleDouble.self, forKey: .costPerHr)
        adjustedCostPerHr = try container.decodeIfPresent(RunPodFlexibleDouble.self, forKey: .adjustedCostPerHr)
        gpu = try container.decodeIfPresent(RunPodGPU.self, forKey: .gpu)
        machine = try container.decodeIfPresent(RunPodMachine.self, forKey: .machine)
        gpuCount = try container.decodeIfPresent(Int.self, forKey: .gpuCount)
        memoryInGb = try container.decodeIfPresent(Double.self, forKey: .memoryInGb)
        publicIp = try container.decodeIfPresent(String.self, forKey: .publicIp)
        portMappings = (try? container.decode([String: Int].self, forKey: .portMappings)) ?? [:]
        ports = (try? container.decode([String].self, forKey: .ports)) ?? []
        templateId = try container.decodeIfPresent(String.self, forKey: .templateId)
        lastStartedAt = try container.decodeIfPresent(String.self, forKey: .lastStartedAt)
        lastStatusChange = try container.decodeIfPresent(String.self, forKey: .lastStatusChange)
        containerDiskInGb = try container.decodeIfPresent(Int.self, forKey: .containerDiskInGb)
        volumeInGb = try container.decodeIfPresent(Int.self, forKey: .volumeInGb)
        volumeMountPath = try container.decodeIfPresent(String.self, forKey: .volumeMountPath)
    }

    var displayName: String {
        let trimmed = (name ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? id : trimmed
    }

    var statusLabel: String {
        desiredStatus ?? "UNKNOWN"
    }

    var gpuLabel: String {
        gpu?.displayName ?? machine?.gpuDisplayName ?? machine?.gpuType?.displayName ?? "GPU"
    }

    var hourlyCostLabel: String? {
        let value = adjustedCostPerHr?.value ?? costPerHr?.value
        guard let value else { return nil }
        return String(format: "$%.2f/hr", value)
    }

    var sshPort: Int? {
        portMappings["22"]
    }

    func sshCommand(sshKeyPath: String) -> String? {
        guard let publicIp, !publicIp.isEmpty, let sshPort else { return nil }
        let key = sshKeyPath.trimmingCharacters(in: .whitespacesAndNewlines)
        let keyArgument = key.isEmpty ? "" : " -i \(shellPathQuoted(key))"
        return "ssh -p \(sshPort)\(keyArgument) root@\(publicIp)"
    }

    func uploadCommand(archiveURL: URL?, sshKeyPath: String) -> String? {
        guard let archiveURL,
              let publicIp,
              !publicIp.isEmpty,
              let sshPort else { return nil }
        let key = sshKeyPath.trimmingCharacters(in: .whitespacesAndNewlines)
        let keyArgument = key.isEmpty ? "" : " -i \(shellPathQuoted(key))"
        let makeJobsDirectory = "ssh -p \(sshPort)\(keyArgument) root@\(publicIp) 'mkdir -p /workspace/jobs'"
        let uploadArchive = "scp -P \(sshPort)\(keyArgument) \(shellPathQuoted(archiveURL.path(percentEncoded: false))) root@\(publicIp):/workspace/jobs/"
        return "\(makeJobsDirectory) && \(uploadArchive)"
    }

    func remotePackageCommand(archiveURL: URL?) -> String? {
        guard let archiveURL else { return nil }
        let zipName = shellQuoted(archiveURL.lastPathComponent)
        return """
        cd /workspace/jobs
        ZIP=\(zipName)
        mkdir -p /workspace/pimp-remote-jobs/current
        if command -v unzip >/dev/null 2>&1; then
          unzip -oq "$ZIP" -d /workspace/pimp-remote-jobs/current
        elif command -v python3 >/dev/null 2>&1; then
          python3 -m zipfile -e "$ZIP" /workspace/pimp-remote-jobs/current
        else
          echo "Install unzip or python3 first."; exit 3
        fi
        cd "$(dirname "$(find /workspace/pimp-remote-jobs/current -maxdepth 4 -name remote_job.json -print -quit)")"
        bash run_remote_training.sh .
        """
    }

    var diagnosticsSummary: String {
        [
            "id=\(id)",
            "name=\(displayName)",
            "status=\(statusLabel)",
            "gpu=\(gpuLabel)",
            "cost=\(hourlyCostLabel ?? "unknown")",
            "ssh=\(sshPort == nil ? "unavailable" : "available")"
        ].joined(separator: " ")
    }

}

struct RunPodGPU: Decodable, Equatable, Sendable {
    var id: String?
    var count: Int?
    var displayName: String?
    var securePrice: RunPodFlexibleDouble?
    var secureSpotPrice: RunPodFlexibleDouble?
}

struct RunPodGPUType: Identifiable, Decodable, Equatable, Sendable {
    var id: String
    var displayName: String?
    var manufacturer: String?
    var memoryInGb: Int?
    var secureCloud: Bool?
    var secureLowestPrice: RunPodLowestPrice?
    var securePrice: RunPodFlexibleDouble?
    var secureSpotPrice: RunPodFlexibleDouble?
    var maxGpuCount: Int?
    var maxGpuCountSecureCloud: Int?

    var selectionValue: String {
        let trimmedID = id.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmedID.isEmpty {
            return trimmedID
        }
        return displayNameValue
    }

    var displayNameValue: String {
        let trimmed = (displayName ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? id : trimmed
    }

    var menuTitle: String {
        var details: [String] = []
        if let memoryInGb {
            details.append("\(memoryInGb) GB")
        }
        if let liveStockLabel {
            details.append(liveStockLabel)
        }
        if let lowestHourlyPriceLabel {
            details.append("from \(lowestHourlyPriceLabel)")
        }
        if details.isEmpty {
            return displayNameValue
        }
        return "\(displayNameValue) (\(details.joined(separator: ", ")))"
    }

    func detailLabel(interruptible: Bool) -> String {
        var details: [String] = []
        if let memoryInGb {
            details.append("\(memoryInGb) GB VRAM")
        }
        details.append(availabilityLabel)
        details.append(stockLabel())
        if let launchPriceLabel = launchPriceLabel(interruptible: interruptible) {
            details.append("\(interruptible ? "Spot" : "On-demand") \(launchPriceLabel)")
        }
        if let maxGpuCount = maxGpuCountSecureCloud ?? maxGpuCount, maxGpuCount > 0 {
            details.append("up to \(maxGpuCount) GPU\(maxGpuCount == 1 ? "" : "s")")
        }
        return details.joined(separator: " · ")
    }

    var availabilityLabel: String {
        if secureCloud == false {
            return "Secure Cloud unavailable"
        }
        return "Secure Cloud"
    }

    var liveStockLabel: String? {
        secureLowestPrice?.stockStatusLabel
    }

    var lowestHourlyPriceLabel: String? {
        let prices = [
            secureLowestPrice?.minimumBidPrice?.value,
            secureLowestPrice?.uninterruptablePrice?.value,
            securePrice?.value,
            secureSpotPrice?.value,
        ].compactMap { $0 }.filter { $0 > 0 }

        guard let lowest = prices.min() else { return nil }
        return String(format: "$%.2f/hr", lowest)
    }

    func matches(_ value: String) -> Bool {
        let normalized = value.trimmingCharacters(in: .whitespacesAndNewlines)
        return normalized == id || normalized == displayNameValue
    }

    private func stockLabel() -> String {
        guard let price = secureLowestPrice else {
            return "Live stock unknown"
        }

        var parts = [price.stockStatusLabel ?? "Live stock unknown"]
        if let countLabel = price.availableGpuCountsLabel {
            parts.append(countLabel)
        }
        return parts.joined(separator: " · ")
    }

    private func launchPriceLabel(interruptible: Bool) -> String? {
        if let price = secureLowestPrice?.hourlyPriceLabel(interruptible: interruptible) {
            return price
        }

        let fallback = interruptible ? secureSpotPrice ?? securePrice : securePrice
        guard let value = fallback?.value, value > 0 else { return nil }
        return String(format: "$%.2f/hr", value)
    }
}

struct RunPodTemplate: Identifiable, Decodable, Equatable, Sendable {
    var id: String
    var name: String?
    var imageName: String?
    var category: String?
    var isRunpod: Bool?
    var isPublic: Bool?
    var isServerless: Bool?
    var containerDiskInGb: Int?
    var volumeInGb: Int?
    var volumeMountPath: String?
    var ports: [String]?

    var displayNameValue: String {
        let trimmed = (name ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? id : trimmed
    }

    var imageNameValue: String? {
        let trimmed = (imageName ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }

    var isOfficialPodTemplate: Bool {
        isRunpod == true && isServerless != true
    }

    var summaryLabel: String {
        if let imageNameValue {
            return "\(displayNameValue) · \(imageNameValue)"
        }
        return displayNameValue
    }

    var detailLabel: String {
        var details: [String] = []
        if isRunpod == true {
            details.append("Official RunPod")
        }
        if let containerDiskInGb {
            details.append("container \(containerDiskInGb) GB")
        }
        if let volumeInGb {
            details.append("volume \(volumeInGb) GB")
        }
        if let ports, !ports.isEmpty {
            details.append(ports.joined(separator: ", "))
        }
        return details.joined(separator: " · ")
    }

    func matchesTemplateID(_ value: String) -> Bool {
        id == value.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    static func recommendedTrainingTemplate(
        for gpuTypeValue: String,
        from templates: [RunPodTemplate]
    ) -> RunPodTemplate? {
        guard supportsNVIDIATrainingTemplate(gpuTypeValue) else { return nil }

        return templates
            .filter { $0.isOfficialPodTemplate }
            .filter { $0.trainingTemplateScore > 0 }
            .sorted { lhs, rhs in
                if lhs.trainingTemplateScore != rhs.trainingTemplateScore {
                    return lhs.trainingTemplateScore > rhs.trainingTemplateScore
                }
                return lhs.displayNameValue.localizedStandardCompare(rhs.displayNameValue) == .orderedAscending
            }
            .first
    }

    private static func supportsNVIDIATrainingTemplate(_ gpuTypeValue: String) -> Bool {
        let normalized = gpuTypeValue.lowercased()
        guard !normalized.contains("amd") && !normalized.contains("mi300") else {
            return false
        }
        return normalized.contains("nvidia")
            || normalized.contains("rtx")
            || normalized.contains("geforce")
            || normalized.contains("a100")
            || normalized.contains("h100")
            || normalized.contains("h200")
            || normalized.contains("l40")
            || normalized.contains("l4")
            || normalized.contains("tesla")
    }

    private var trainingTemplateScore: Int {
        let searchable = "\(displayNameValue) \(imageNameValue ?? "")".lowercased()
        guard searchable.contains("pytorch") else { return 0 }

        var score = 1_000
        if searchable.contains("runpod/pytorch") { score += 500 }
        if searchable.contains("runpod pytorch") { score += 500 }
        if searchable.contains("cuda") { score += 250 }
        if searchable.contains("cuda12") { score += 250 }
        if searchable.contains("cuda12.4") { score += 400 }
        if searchable.contains("devel") { score += 150 }
        if searchable.contains("py3.11") { score += 100 }
        if searchable.contains("ubuntu22.04") { score += 100 }
        if searchable.contains("2.4.0") { score += 500 }
        if searchable.contains("serverless") { score -= 1_000 }
        return score
    }
}

struct RunPodLowestPrice: Decodable, Equatable, Sendable {
    var gpuName: String?
    var gpuTypeId: String?
    var minimumBidPrice: RunPodFlexibleDouble?
    var uninterruptablePrice: RunPodFlexibleDouble?
    var stockStatus: String?
    var availableGpuCounts: [Int]?
    var maxGpuCount: Int?
    var maxUnreservedGpuCount: Int?
    var rentedCount: Int?
    var totalCount: Int?

    var stockStatusLabel: String? {
        let trimmed = (stockStatus ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        if trimmed.caseInsensitiveCompare("none") == .orderedSame {
            return "No live stock"
        }
        return "\(trimmed.capitalized) stock"
    }

    var availableGpuCountsLabel: String? {
        let counts = (availableGpuCounts ?? []).filter { $0 > 0 }.sorted()
        guard !counts.isEmpty else { return nil }
        let joined = counts.map(String.init).joined(separator: "/")
        return "counts \(joined)"
    }

    func hourlyPriceLabel(interruptible: Bool) -> String? {
        let price = interruptible
            ? minimumBidPrice?.value ?? uninterruptablePrice?.value
            : uninterruptablePrice?.value ?? minimumBidPrice?.value
        guard let price, price > 0 else { return nil }
        return String(format: "$%.2f/hr", price)
    }
}

struct RunPodMachine: Decodable, Equatable, Sendable {
    var gpuTypeId: String?
    var gpuDisplayName: String?
    var location: String?
    var dataCenterId: String?
    var gpuType: RunPodGPU?
}

struct RunPodFlexibleDouble: Codable, Equatable, Sendable {
    var value: Double

    init(_ value: Double) {
        self.value = value
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let double = try? container.decode(Double.self) {
            value = double
            return
        }
        if let int = try? container.decode(Int.self) {
            value = Double(int)
            return
        }
        if let string = try? container.decode(String.self),
           let double = Double(string) {
            value = double
            return
        }
        throw DecodingError.dataCorruptedError(
            in: container,
            debugDescription: "Expected numeric or string double value"
        )
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(value)
    }
}

struct RunPodCreatePodRequest: Encodable, Equatable, Sendable {
    var cloudType: String
    var computeType: String
    var containerDiskInGb: Int
    var gpuCount: Int
    var gpuTypeIds: [String]
    var gpuTypePriority: String
    var imageName: String?
    var interruptible: Bool
    var locked: Bool
    var name: String
    var ports: [String]
    var supportPublicIp: Bool
    var templateId: String?
    var volumeInGb: Int
    var volumeMountPath: String
}

private struct RunPodGraphQLRequest: Encodable {
    var query: String
}

private struct RunPodGraphQLResponse<Payload: Decodable>: Decodable {
    var data: Payload?
    var errors: [RunPodGraphQLError]?
}

private struct RunPodGraphQLError: Decodable, Equatable {
    var message: String
}

private struct RunPodGPUTypesPayload: Decodable {
    var gpuTypes: [RunPodGPUType]
}

protocol RunPodTransport: Sendable {
    func data(for request: URLRequest) async throws -> (Data, HTTPURLResponse)
}

struct URLSessionRunPodTransport: RunPodTransport {
    var session: URLSession = .shared

    func data(for request: URLRequest) async throws -> (Data, HTTPURLResponse) {
        let (data, response) = try await session.data(for: request)
        guard let httpResponse = response as? HTTPURLResponse else {
            throw RunPodError.invalidResponse
        }
        return (data, httpResponse)
    }
}

struct RunPodAPIClient {
    var baseURL: URL
    var graphQLURL: URL
    var transport: any RunPodTransport

    init(
        baseURL: URL = URL(string: "https://rest.runpod.io/v1")!,
        graphQLURL: URL = URL(string: "https://api.runpod.io/graphql")!,
        transport: any RunPodTransport = URLSessionRunPodTransport()
    ) {
        self.baseURL = baseURL
        self.graphQLURL = graphQLURL
        self.transport = transport
    }

    func listPods(token: String) async throws -> [RunPodPodSummary] {
        let request = try makeRequest(path: "pods", method: "GET", token: token)
        let (data, response) = try await transport.data(for: request)
        try validate(response: response, data: data)
        return try JSONDecoder().decode([RunPodPodSummary].self, from: data)
    }

    func listGPUTypes(token: String) async throws -> [RunPodGPUType] {
        let body = try JSONEncoder().encode(RunPodGraphQLRequest(query: Self.gpuTypesQuery))
        let request = try makeGraphQLRequest(token: token, body: body)
        let (data, response) = try await transport.data(for: request)
        try validate(response: response, data: data)

        let payload = try JSONDecoder().decode(RunPodGraphQLResponse<RunPodGPUTypesPayload>.self, from: data)
        if let errors = payload.errors, !errors.isEmpty {
            throw RunPodError.graphQLErrors(errors.map(\.message))
        }
        guard let data = payload.data else {
            throw RunPodError.invalidResponse
        }

        return data.gpuTypes
            .filter { $0.secureCloud != false }
            .sorted {
                $0.displayNameValue.localizedStandardCompare($1.displayNameValue) == .orderedAscending
            }
    }

    func listOfficialPodTemplates(token: String) async throws -> [RunPodTemplate] {
        let request = try makeRequest(
            path: "templates",
            method: "GET",
            token: token,
            queryItems: [
                URLQueryItem(name: "includeRunpodTemplates", value: "true"),
                URLQueryItem(name: "includePublicTemplates", value: "false"),
                URLQueryItem(name: "includeEndpointBoundTemplates", value: "false"),
            ]
        )
        let (data, response) = try await transport.data(for: request)
        try validate(response: response, data: data)
        return try JSONDecoder().decode([RunPodTemplate].self, from: data)
            .filter(\.isOfficialPodTemplate)
            .sorted {
                $0.displayNameValue.localizedStandardCompare($1.displayNameValue) == .orderedAscending
            }
    }

    func createPod(token: String, configuration: RunPodLaunchConfiguration) async throws -> RunPodPodSummary {
        let body = try JSONEncoder().encode(configuration.requestBody())
        let request = try makeRequest(path: "pods", method: "POST", token: token, body: body)
        let (data, response) = try await transport.data(for: request)
        try validate(response: response, data: data)
        return try JSONDecoder().decode(RunPodPodSummary.self, from: data)
    }

    func deletePod(token: String, podID: String) async throws {
        let encodedID = podID.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? podID
        let request = try makeRequest(path: "pods/\(encodedID)", method: "DELETE", token: token)
        let (data, response) = try await transport.data(for: request)
        try validate(response: response, data: data)
    }

    private func makeRequest(
        path: String,
        method: String,
        token: String,
        queryItems: [URLQueryItem] = [],
        body: Data? = nil
    ) throws -> URLRequest {
        let trimmedToken = token.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedToken.isEmpty else {
            throw RunPodError.missingAPIKey
        }

        let base = baseURL.appendingPathComponent(path)
        var components = URLComponents(url: base, resolvingAgainstBaseURL: false)
        if !queryItems.isEmpty {
            components?.queryItems = queryItems
        }
        guard let url = components?.url else {
            throw RunPodError.invalidResponse
        }
        var request = URLRequest(url: url)
        request.httpMethod = method
        request.setValue("Bearer \(trimmedToken)", forHTTPHeaderField: "Authorization")
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        if let body {
            request.httpBody = body
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        }
        return request
    }

    private func makeGraphQLRequest(token: String, body: Data) throws -> URLRequest {
        let trimmedToken = token.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedToken.isEmpty else {
            throw RunPodError.missingAPIKey
        }

        var request = URLRequest(url: graphQLURL)
        request.httpMethod = "POST"
        request.httpBody = body
        request.setValue("Bearer \(trimmedToken)", forHTTPHeaderField: "Authorization")
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        return request
    }

    private func validate(response: HTTPURLResponse, data: Data) throws {
        guard (200..<300).contains(response.statusCode) else {
            let body = String(data: data, encoding: .utf8)?
                .trimmingCharacters(in: .whitespacesAndNewlines)
            throw RunPodError.httpStatus(response.statusCode, body)
        }
    }

    private static let gpuTypesQuery = """
    query PiMPAudioRunPodGPUTypes {
      gpuTypes {
        id
        displayName
        manufacturer
        memoryInGb
        secureCloud
        secureLowestPrice: lowestPrice(input: { gpuCount: 1, secureCloud: true }) {
          ...RunPodLowestPriceFields
        }
        securePrice
        secureSpotPrice
        maxGpuCount
        maxGpuCountSecureCloud
      }
    }

    fragment RunPodLowestPriceFields on LowestPrice {
      gpuName
      gpuTypeId
      minimumBidPrice
      uninterruptablePrice
      stockStatus
      availableGpuCounts
      maxGpuCount
      maxUnreservedGpuCount
      rentedCount
      totalCount
    }
    """
}

enum RunPodError: LocalizedError, Equatable {
    case missingAPIKey
    case invalidConfiguration(String)
    case invalidResponse
    case graphQLErrors([String])
    case httpStatus(Int, String?)

    var errorDescription: String? {
        switch self {
        case .missingAPIKey:
            return "RunPod API key is not configured."
        case .invalidConfiguration(let message):
            return message
        case .invalidResponse:
            return "RunPod returned a response PiMP Audio Studio could not read."
        case .graphQLErrors(let messages):
            return "RunPod rejected the GPU type request: \(Self.joinedDetails(messages))"
        case .httpStatus(let status, let body):
            return Self.message(forHTTPStatus: status, body: body)
        }
    }

    static func userFacingMessage(for error: Error) -> String {
        if let runPodError = error as? RunPodError {
            return runPodError.localizedDescription
        }

        if let urlError = error as? URLError {
            switch urlError.code {
            case .notConnectedToInternet,
                 .cannotFindHost,
                 .cannotConnectToHost,
                 .networkConnectionLost,
                 .dnsLookupFailed,
                 .timedOut:
                return "Could not reach RunPod. Check your connection and try again. (\(urlError.localizedDescription))"
            default:
                return "RunPod network request failed: \(urlError.localizedDescription)"
            }
        }

        return error.localizedDescription
    }

    private static func message(forHTTPStatus status: Int, body: String?) -> String {
        let detail = cleanDetail(body)
        switch status {
        case 401:
            return "RunPod rejected the API key. Check that it was copied correctly or generate a new key in RunPod.\(detail)"
        case 403:
            return "RunPod accepted the request but this API key is not authorized for that action. Check the key permissions or account status.\(detail)"
        case 404:
            return "RunPod could not find that resource. Refresh pods and try again.\(detail)"
        case 429:
            return "RunPod rate-limited the request. Wait a minute and try again.\(detail)"
        case 500...599:
            return "RunPod is having trouble right now (HTTP \(status)). Try again shortly.\(detail)"
        default:
            return "RunPod request failed with HTTP \(status).\(detail)"
        }
    }

    private static func cleanDetail(_ body: String?) -> String {
        guard let body, !body.isEmpty else { return "" }
        let trimmed = body.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return "" }
        let capped = trimmed.count > 240 ? "\(trimmed.prefix(240))..." : trimmed
        return " Detail: \(capped)"
    }

    private static func joinedDetails(_ messages: [String]) -> String {
        let clean = messages
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        guard !clean.isEmpty else { return "unknown error" }
        return clean.joined(separator: "; ")
    }
}

private func shellQuoted(_ value: String) -> String {
    "'\(shellSingleQuotedContent(value))'"
}

private func shellPathQuoted(_ value: String) -> String {
    if value.hasPrefix("~/") {
        let suffix = String(value.dropFirst(2))
        if suffix.rangeOfCharacter(from: shellUnsafePathCharacters) == nil {
            return value
        }
        return "~/" + shellQuoted(suffix)
    }
    if value.rangeOfCharacter(from: shellUnsafePathCharacters) == nil {
        return value
    }
    return shellQuoted(value)
}

private func shellSingleQuotedContent(_ value: String) -> String {
    value.replacingOccurrences(of: "'", with: "'\\''")
}

private let shellUnsafePathCharacters = CharacterSet.whitespacesAndNewlines
    .union(CharacterSet(charactersIn: "'\"\\$`!()[]{};&|<>*?"))
