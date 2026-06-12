import Foundation
import SwiftUI

@MainActor
final class RunPodService: ObservableObject {
    @Published private(set) var pods: [RunPodPodSummary] = []
    @Published private(set) var isRefreshing = false
    @Published private(set) var isLaunching = false
    @Published private(set) var stoppingPodID: String?
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
        lastMessage = "RunPod API key saved."
        lastError = nil
    }

    func removeAPIKey() {
        apiKeyStore.removeAPIKey(for: APIKeyStore.runPodProviderID)
        pods = []
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
            pods = try await client.listPods(token: token)
            lastMessage = "Loaded \(pods.count) RunPod pod\(pods.count == 1 ? "" : "s")."
            lastError = nil
        } catch {
            lastError = error.localizedDescription
        }
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
            let pod = try await client.createPod(token: token, configuration: configuration)
            upsert(pod)
            lastMessage = "RunPod launch requested for \(pod.displayName)."
            lastError = nil
            return pod
        } catch {
            lastError = error.localizedDescription
            return nil
        }
    }

    func stopPod(id: String) async {
        guard let token = runPodToken() else {
            lastError = "Add a RunPod API key before stopping a pod."
            return
        }

        stoppingPodID = id
        defer { stoppingPodID = nil }

        do {
            try await client.stopPod(token: token, podID: id)
            lastMessage = "Stop requested for RunPod pod \(id)."
            lastError = nil
            await refreshPods()
        } catch {
            lastError = error.localizedDescription
        }
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
}

struct RunPodLaunchConfiguration: Equatable, Sendable {
    var name: String
    var templateID: String
    var imageName: String
    var gpuTypeID: String
    var cloudType: RunPodCloudType
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

        return RunPodCreatePodRequest(
            cloudType: cloudType.rawValue,
            computeType: "GPU",
            containerDiskInGb: containerDiskGB,
            gpuCount: 1,
            gpuTypeIds: [trimmedGPUType],
            gpuTypePriority: "availability",
            imageName: trimmedImageName.isEmpty ? nil : trimmedImageName,
            interruptible: interruptible,
            locked: false,
            name: trimmedName.isEmpty ? "PiMP Audio Studio Remote Job" : trimmedName,
            ports: ports,
            supportPublicIp: supportPublicIP,
            templateId: trimmedTemplateID.isEmpty ? nil : trimmedTemplateID,
            volumeInGb: volumeGB,
            volumeMountPath: "/workspace"
        )
    }
}

enum RunPodCloudType: String, CaseIterable, Identifiable, Codable, Sendable {
    case secure = "SECURE"
    case community = "COMMUNITY"

    var id: String { rawValue }

    var title: String {
        switch self {
        case .secure: return "Secure"
        case .community: return "Community"
        }
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
        return "ssh root@\(publicIp) -p \(sshPort)\(keyArgument)"
    }

    func uploadCommand(archiveURL: URL?, sshKeyPath: String) -> String? {
        guard let archiveURL,
              let publicIp,
              !publicIp.isEmpty,
              let sshPort else { return nil }
        let key = sshKeyPath.trimmingCharacters(in: .whitespacesAndNewlines)
        let keyArgument = key.isEmpty ? "" : " -i \(shellPathQuoted(key))"
        return "scp -P \(sshPort)\(keyArgument) \(shellPathQuoted(archiveURL.path(percentEncoded: false))) root@\(publicIp):/workspace/jobs/"
    }

}

struct RunPodGPU: Decodable, Equatable, Sendable {
    var id: String?
    var count: Int?
    var displayName: String?
    var securePrice: RunPodFlexibleDouble?
    var communityPrice: RunPodFlexibleDouble?
    var secureSpotPrice: RunPodFlexibleDouble?
    var communitySpotPrice: RunPodFlexibleDouble?
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
    var transport: any RunPodTransport

    init(
        baseURL: URL = URL(string: "https://rest.runpod.io/v1")!,
        transport: any RunPodTransport = URLSessionRunPodTransport()
    ) {
        self.baseURL = baseURL
        self.transport = transport
    }

    func listPods(token: String) async throws -> [RunPodPodSummary] {
        let request = try makeRequest(path: "pods", method: "GET", token: token)
        let (data, response) = try await transport.data(for: request)
        try validate(response: response, data: data)
        return try JSONDecoder().decode([RunPodPodSummary].self, from: data)
    }

    func createPod(token: String, configuration: RunPodLaunchConfiguration) async throws -> RunPodPodSummary {
        let body = try JSONEncoder().encode(configuration.requestBody())
        let request = try makeRequest(path: "pods", method: "POST", token: token, body: body)
        let (data, response) = try await transport.data(for: request)
        try validate(response: response, data: data)
        return try JSONDecoder().decode(RunPodPodSummary.self, from: data)
    }

    func stopPod(token: String, podID: String) async throws {
        let encodedID = podID.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? podID
        let request = try makeRequest(path: "pods/\(encodedID)/stop", method: "POST", token: token)
        let (data, response) = try await transport.data(for: request)
        try validate(response: response, data: data)
    }

    private func makeRequest(
        path: String,
        method: String,
        token: String,
        body: Data? = nil
    ) throws -> URLRequest {
        let trimmedToken = token.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedToken.isEmpty else {
            throw RunPodError.missingAPIKey
        }

        let url = baseURL.appendingPathComponent(path)
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

    private func validate(response: HTTPURLResponse, data: Data) throws {
        guard (200..<300).contains(response.statusCode) else {
            let body = String(data: data, encoding: .utf8)?
                .trimmingCharacters(in: .whitespacesAndNewlines)
            throw RunPodError.httpStatus(response.statusCode, body)
        }
    }
}

enum RunPodError: LocalizedError, Equatable {
    case missingAPIKey
    case invalidConfiguration(String)
    case invalidResponse
    case httpStatus(Int, String?)

    var errorDescription: String? {
        switch self {
        case .missingAPIKey:
            return "RunPod API key is not configured."
        case .invalidConfiguration(let message):
            return message
        case .invalidResponse:
            return "RunPod returned a non-HTTP response."
        case .httpStatus(let status, let body):
            let detail = body?.isEmpty == false ? ": \(body!)" : ""
            return "RunPod request failed with HTTP \(status)\(detail)"
        }
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
