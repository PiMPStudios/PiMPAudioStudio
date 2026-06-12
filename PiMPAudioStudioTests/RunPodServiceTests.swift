import Foundation
import Testing
@testable import PiMPAudioStudio

private enum RunPodMockOutcome {
    case response(statusCode: Int, data: Data)
    case failure(Error)
}

private final class RunPodMockTransport: RunPodTransport, @unchecked Sendable {
    private let lock = NSLock()
    private var outcomes: [RunPodMockOutcome]
    private(set) var requests: [URLRequest] = []

    init(responses: [(statusCode: Int, data: Data)]) {
        self.outcomes = responses.map { .response(statusCode: $0.statusCode, data: $0.data) }
    }

    init(outcomes: [RunPodMockOutcome]) {
        self.outcomes = outcomes
    }

    func data(for request: URLRequest) async throws -> (Data, HTTPURLResponse) {
        lock.lock()
        requests.append(request)
        let outcome = outcomes.isEmpty
            ? .response(statusCode: 200, data: Data("{}".utf8))
            : outcomes.removeFirst()
        lock.unlock()

        let response: (statusCode: Int, data: Data)
        switch outcome {
        case .response(let statusCode, let data):
            response = (statusCode, data)
        case .failure(let error):
            throw error
        }

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

@Test func runPodClientListGPUTypesParsesPricingAndAvailability() async throws {
    let transport = RunPodMockTransport(responses: [(
        statusCode: 200,
        data: Data("""
        {
          "data": {
            "gpuTypes": [
              {
                "id": "NVIDIA GeForce RTX 4090",
                "displayName": "NVIDIA GeForce RTX 4090",
                "manufacturer": "NVIDIA",
                "memoryInGb": 24,
                "secureCloud": true,
                "secureLowestPrice": {
                  "minimumBidPrice": 0.50,
                  "uninterruptablePrice": 0.74,
                  "stockStatus": "Medium",
                  "availableGpuCounts": [1, 2]
                },
                "securePrice": 0.74,
                "secureSpotPrice": 0.50,
                "maxGpuCount": 8,
                "maxGpuCountSecureCloud": 4
              },
              {
                "id": "NVIDIA H100 PCIe",
                "displayName": "NVIDIA H100 PCIe",
                "memoryInGb": 80,
                "secureCloud": true,
                "secureLowestPrice": {
                  "uninterruptablePrice": 4.25,
                  "stockStatus": "None",
                  "availableGpuCounts": []
                },
                "securePrice": 4.25,
                "maxGpuCount": 2
              },
              {
                "id": "community-only-gpu",
                "displayName": "Community Only GPU",
                "memoryInGb": 24,
                "secureCloud": false
              }
            ]
          }
        }
        """.utf8)
    )])
    let client = RunPodAPIClient(
        baseURL: URL(string: "https://rest.runpod.io/v1")!,
        graphQLURL: URL(string: "https://api.runpod.io/graphql")!,
        transport: transport
    )

    let gpuTypes = try await client.listGPUTypes(token: "rp_test_token")

    #expect(gpuTypes.count == 2)
    #expect(!gpuTypes.contains { $0.id == "community-only-gpu" })
    let rtx4090 = try #require(gpuTypes.first { $0.id == "NVIDIA GeForce RTX 4090" })
    #expect(rtx4090.selectionValue == "NVIDIA GeForce RTX 4090")
    #expect(rtx4090.menuTitle.contains("24 GB"))
    #expect(rtx4090.menuTitle.contains("Medium stock"))
    #expect(rtx4090.menuTitle.contains("from $0.50/hr"))
    let secureDetail = rtx4090.detailLabel(interruptible: false)
    #expect(secureDetail.contains("Secure Cloud"))
    #expect(secureDetail.contains("Medium stock"))
    #expect(secureDetail.contains("counts 1/2"))
    #expect(secureDetail.contains("On-demand $0.74/hr"))
    #expect(secureDetail.contains("up to 4 GPUs"))

    let secureSpotDetail = rtx4090.detailLabel(interruptible: true)
    #expect(secureSpotDetail.contains("Medium stock"))
    #expect(secureSpotDetail.contains("Spot $0.50/hr"))

    let request = try #require(transport.requests.first)
    #expect(request.httpMethod == "POST")
    #expect(request.url?.absoluteString == "https://api.runpod.io/graphql")
    #expect(request.value(forHTTPHeaderField: "Authorization") == "Bearer rp_test_token")
    let bodyData = try #require(request.httpBody)
    let body = String(data: bodyData, encoding: .utf8) ?? ""
    #expect(body.contains("gpuTypes"))
    #expect(body.contains("lowestPrice"))
    #expect(body.contains("stockStatus"))
    #expect(!body.contains("rp_test_token"))
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

@Test func runPodClientInvalidAPIKeyErrorIsSpecific() async throws {
    let transport = RunPodMockTransport(responses: [(
        statusCode: 401,
        data: Data(#"{"error":"JWT expired"}"#.utf8)
    )])
    let client = RunPodAPIClient(
        baseURL: URL(string: "https://rest.runpod.io/v1")!,
        transport: transport
    )

    do {
        _ = try await client.listPods(token: "expired-token")
        Issue.record("Expected invalid API key error")
    } catch {
        #expect(error.localizedDescription.contains("RunPod rejected the API key"))
        #expect(error.localizedDescription.contains("generate a new key"))
    }
}

@Test func runPodGraphQLErrorsAreReadable() async throws {
    let transport = RunPodMockTransport(responses: [(
        statusCode: 200,
        data: Data("""
        {
          "errors": [
            {"message": "Unauthorized"}
          ]
        }
        """.utf8)
    )])
    let client = RunPodAPIClient(transport: transport)

    do {
        _ = try await client.listGPUTypes(token: "bad-token")
        Issue.record("Expected GraphQL error")
    } catch {
        #expect(error.localizedDescription.contains("RunPod rejected the GPU type request"))
        #expect(error.localizedDescription.contains("Unauthorized"))
    }
}

@Test func runPodNetworkErrorsAreReadable() {
    let message = RunPodError.userFacingMessage(for: URLError(.timedOut))

    #expect(message.contains("Could not reach RunPod"))
    #expect(message.contains("try again"))
}
