import Foundation
import Testing
@testable import PiMPAudioStudio

private final class RunPodMockTransport: RunPodTransport, @unchecked Sendable {
    private let lock = NSLock()
    private var responses: [(statusCode: Int, data: Data)]
    private(set) var requests: [URLRequest] = []

    init(responses: [(statusCode: Int, data: Data)]) {
        self.responses = responses
    }

    func data(for request: URLRequest) async throws -> (Data, HTTPURLResponse) {
        lock.lock()
        requests.append(request)
        let response = responses.isEmpty
            ? (statusCode: 200, data: Data("{}".utf8))
            : responses.removeFirst()
        lock.unlock()

        let httpResponse = HTTPURLResponse(
            url: request.url!,
            statusCode: response.statusCode,
            httpVersion: nil,
            headerFields: nil
        )!
        return (response.data, httpResponse)
    }
}

@Test func apiKeyStoreKeepsRunPodOutOfLyricProviderList() {
    #expect(APIKeyStore.apiKeyProviders.contains { $0.id == APIKeyStore.runPodProviderID })
    #expect(!APIKeyStore.providers.contains { $0.id == APIKeyStore.runPodProviderID })
}

@Test func runPodClientListPodsParsesPodStatusAndSSHDetails() async throws {
    let transport = RunPodMockTransport(responses: [(
        statusCode: 200,
        data: Data("""
        [
          {
            "id": "abc123",
            "name": "pimp-training",
            "desiredStatus": "RUNNING",
            "costPerHr": "0.74",
            "adjustedCostPerHr": 0.69,
            "gpu": {"displayName": "NVIDIA RTX PRO 6000 Blackwell Server Edition"},
            "publicIp": "216.81.151.42",
            "portMappings": {"22": 11646},
            "ports": ["22/tcp", "8888/http"]
          }
        ]
        """.utf8)
    )])
    let client = RunPodAPIClient(
        baseURL: URL(string: "https://rest.runpod.io/v1")!,
        transport: transport
    )

    let pods = try await client.listPods(token: "rp_test_token")

    #expect(pods.count == 1)
    #expect(pods[0].id == "abc123")
    #expect(pods[0].statusLabel == "RUNNING")
    #expect(pods[0].hourlyCostLabel == "$0.69/hr")
    #expect(pods[0].sshPort == 11646)
    #expect(pods[0].sshCommand(sshKeyPath: "~/.ssh/id_ed25519") == "ssh root@216.81.151.42 -p 11646 -i ~/.ssh/id_ed25519")

    #expect(transport.requests.first?.httpMethod == "GET")
    #expect(transport.requests.first?.url?.absoluteString == "https://rest.runpod.io/v1/pods")
    #expect(transport.requests.first?.value(forHTTPHeaderField: "Authorization") == "Bearer rp_test_token")
}

@Test func runPodClientCreatePodBuildsBearerRequestWithoutLeakingTokenIntoBody() async throws {
    let transport = RunPodMockTransport(responses: [(
        statusCode: 201,
        data: Data("""
        {
          "id": "newpod",
          "name": "pimp-remote",
          "desiredStatus": "RUNNING",
          "portMappings": {}
        }
        """.utf8)
    )])
    let client = RunPodAPIClient(
        baseURL: URL(string: "https://rest.runpod.io/v1")!,
        transport: transport
    )
    let configuration = RunPodLaunchConfiguration(
        name: "pimp-remote",
        templateID: "template-123",
        imageName: "",
        gpuTypeID: "NVIDIA GeForce RTX 4090",
        cloudType: .secure,
        interruptible: false,
        containerDiskGB: 80,
        volumeGB: 120
    )

    let pod = try await client.createPod(token: "secret-token", configuration: configuration)

    #expect(pod.id == "newpod")
    let request = try #require(transport.requests.first)
    #expect(request.httpMethod == "POST")
    #expect(request.url?.absoluteString == "https://rest.runpod.io/v1/pods")
    #expect(request.value(forHTTPHeaderField: "Authorization") == "Bearer secret-token")
    #expect(request.value(forHTTPHeaderField: "Content-Type") == "application/json")

    let bodyData = try #require(request.httpBody)
    let body = try #require(JSONSerialization.jsonObject(with: bodyData) as? [String: Any])
    #expect(body["templateId"] as? String == "template-123")
    #expect(body["computeType"] as? String == "GPU")
    #expect(body["cloudType"] as? String == "SECURE")
    #expect(body["gpuTypeIds"] as? [String] == ["NVIDIA GeForce RTX 4090"])
    #expect(body["volumeInGb"] as? Int == 120)
    #expect(!String(data: bodyData, encoding: .utf8)!.contains("secret-token"))
}

@Test func runPodClientStopPodUsesStopEndpoint() async throws {
    let transport = RunPodMockTransport(responses: [(
        statusCode: 200,
        data: Data()
    )])
    let client = RunPodAPIClient(
        baseURL: URL(string: "https://rest.runpod.io/v1")!,
        transport: transport
    )

    try await client.stopPod(token: "rp_test_token", podID: "abc123")

    let request = try #require(transport.requests.first)
    #expect(request.httpMethod == "POST")
    #expect(request.url?.absoluteString == "https://rest.runpod.io/v1/pods/abc123/stop")
    #expect(request.value(forHTTPHeaderField: "Authorization") == "Bearer rp_test_token")
}
