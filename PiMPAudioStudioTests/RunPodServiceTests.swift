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

private actor RunPodRemoteMockExecutor: RunPodRemoteCommandExecuting {
    private var results: [RunPodRemoteCommandResult]
    private(set) var invocations: [(URL, [String])] = []

    init(results: [RunPodRemoteCommandResult]) {
        self.results = results
    }

    func run(
        executableURL: URL,
        arguments: [String],
        timeoutSeconds: TimeInterval
    ) async throws -> RunPodRemoteCommandResult {
        invocations.append((executableURL, arguments))
        if results.isEmpty {
            return RunPodRemoteCommandResult(
                exitStatus: 0,
                stdout: "",
                stderr: ""
            )
        }
        return results.removeFirst()
    }
}

private func runPodTestPod() throws -> RunPodPodSummary {
    try JSONDecoder().decode(
        RunPodPodSummary.self,
        from: Data(
            """
            {
              "id": "pod-test-123",
              "name": "pimp-test",
              "desiredStatus": "RUNNING",
              "publicIp": "216.81.151.42",
              "portMappings": {"22": 11646},
              "ports": ["22/tcp"]
            }
            """.utf8
        )
    )
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
    #expect(pods[0].sshCommand(sshKeyPath: "~/.ssh/id_ed25519") == "ssh -p 11646 -i ~/.ssh/id_ed25519 root@216.81.151.42")

    let archiveURL = URL(filePath: "/Users/test/Training Jobs/job one.zip")
    let workerURL = URL(filePath: "/Users/test/PiMPAudioStudio/remote-worker")
    let uploadCommand = try #require(pods[0].uploadCommand(
        archiveURL: archiveURL,
        sshKeyPath: "~/.ssh/id_ed25519",
        workerDirectoryURL: workerURL
    ))
    #expect(uploadCommand.contains("WORKER_DIR='/Users/test/PiMPAudioStudio/remote-worker'"))
    #expect(uploadCommand.contains("Could not find PiMP remote worker"))
    #expect(uploadCommand.contains("ssh -p 11646 -i ~/.ssh/id_ed25519 root@216.81.151.42 'mkdir -p /workspace/jobs && rm -rf /workspace/pimp-remote-worker-src'"))
    #expect(uploadCommand.contains("scp -P 11646 -i ~/.ssh/id_ed25519 '/Users/test/Training Jobs/job one.zip' root@216.81.151.42:/workspace/jobs/"))
    #expect(uploadCommand.contains("scp -P 11646 -i ~/.ssh/id_ed25519 -r \"$WORKER_DIR\" root@216.81.151.42:/workspace/pimp-remote-worker-src"))

    let remoteCommand = try #require(pods[0].remotePackageCommand(archiveURL: archiveURL))
    #expect(remoteCommand.contains("ZIP='job one.zip'"))
    #expect(remoteCommand.contains("bash /workspace/pimp-remote-worker-src/install.sh"))
    #expect(remoteCommand.contains("source /workspace/pimp-remote-worker/env.sh"))
    #expect(remoteCommand.contains("pimp-remote-worker train \"$ZIP\""))
    #expect(!remoteCommand.contains("unzip -oq"))

    #expect(transport.requests.first?.httpMethod == "GET")
    #expect(transport.requests.first?.url?.absoluteString == "https://rest.runpod.io/v1/pods")
    #expect(transport.requests.first?.value(forHTTPHeaderField: "Authorization") == "Bearer rp_test_token")
}

@Test func runPodRemoteJobPlanUsesArgumentArraysAndScopedRemotePaths() throws {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString, isDirectory: true)
    let workerDirectory = root.appendingPathComponent(
        "remote worker",
        isDirectory: true
    )
    try FileManager.default.createDirectory(
        at: workerDirectory,
        withIntermediateDirectories: true
    )
    defer { try? FileManager.default.removeItem(at: root) }
    let archiveURL = root.appendingPathComponent("Whole Track Job.zip")
    try Data("zip".utf8).write(to: archiveURL)
    try Data("#!/bin/bash".utf8).write(
        to: workerDirectory.appendingPathComponent("install.sh")
    )

    let plan = try RunPodRemoteJobPlan(
        pod: runPodTestPod(),
        archiveURL: archiveURL,
        sshKeyPath: "",
        workerDirectoryURL: workerDirectory
    )

    #expect(plan.remoteJobKey == "whole-track-job")
    #expect(plan.prepareUploadCommand.executableURL.path == "/usr/bin/ssh")
    #expect(plan.uploadArchiveCommand.executableURL.path == "/usr/bin/scp")
    #expect(plan.uploadArchiveCommand.arguments.contains(archiveURL.path))
    #expect(
        plan.uploadWorkerCommand.arguments.contains(
            workerDirectory.path(percentEncoded: false)
        )
    )
    #expect(
        plan.startCommand.arguments.last?
            .contains("/workspace/pimp-remote-jobs/whole-track-job") == true
    )
    #expect(
        plan.startCommand.arguments.last?
            .contains("nohup pimp-remote-worker train") == true
    )
    #expect(
        plan.startCommand.arguments.last?
            .contains("unzip -oq") == false
    )
    #expect(
        plan.startCommand.arguments.last?
            .contains("PIMP_REMOTE_JOB_DIR") == true
    )
    #expect(
        plan.preparePullbackCommand.arguments.last?
            .contains("remote_artifacts/result.json") == true
    )
    #expect(
        !plan.startCommand.arguments
            .joined(separator: " ")
            .localizedCaseInsensitiveContains("api_key")
    )
}

@Test func runPodRemoteStatusParsesStableResultContract() throws {
    let snapshot = try RunPodRemoteStatusSnapshot.parse(
        """
        PIMP_STATUS:RESULT
        {
          "pimp_audio_remote_result_version": "1.0",
          "job_id": "job-123",
          "task": "train_lora",
          "status": "succeeded",
          "started_at": "2026-07-18T00:00:00Z",
          "finished_at": "2026-07-18T00:10:00Z",
          "artifacts": [
            {
              "kind": "final_adapter",
              "path": "remote_artifacts/final_adapter",
              "media_type": "application/vnd.pimp-audio.adapter"
            }
          ],
          "metrics": {"sample_count": 2}
        }
        """
    )

    if case .result = snapshot.status {
        #expect(snapshot.result?.jobID == "job-123")
        #expect(snapshot.result?.status == .succeeded)
        #expect(snapshot.result?.artifacts.first?.kind == "final_adapter")
    } else {
        Issue.record("Expected a parsed result snapshot")
    }
}

@Test func runPodRemoteStatusRejectsUnknownResultSchema() {
    #expect(throws: RunPodRemoteJobError.self) {
        try RunPodRemoteStatusSnapshot.parse(
            """
            PIMP_STATUS:RESULT
            {
              "pimp_audio_remote_result_version": "2.0",
              "job_id": "job-123",
              "task": "train_lora",
              "status": "succeeded",
              "artifacts": [],
              "metrics": {}
            }
            """
        )
    }
}

@MainActor
@Test func runPodServiceUploadsArchiveAndWorkerWithoutShellInterpolation() async throws {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString, isDirectory: true)
    let workerDirectory = root.appendingPathComponent(
        "remote-worker",
        isDirectory: true
    )
    try FileManager.default.createDirectory(
        at: workerDirectory,
        withIntermediateDirectories: true
    )
    defer { try? FileManager.default.removeItem(at: root) }
    let archiveURL = root.appendingPathComponent("job.zip")
    try Data("zip".utf8).write(to: archiveURL)
    try Data("#!/bin/bash".utf8).write(
        to: workerDirectory.appendingPathComponent("install.sh")
    )

    let executor = RunPodRemoteMockExecutor(
        results: Array(
            repeating: RunPodRemoteCommandResult(
                exitStatus: 0,
                stdout: "",
                stderr: ""
            ),
            count: 3
        )
    )
    let service = RunPodService(
        apiKeyStore: APIKeyStore(),
        remoteExecutor: executor
    )
    let pod = try runPodTestPod()

    await service.uploadRemoteJob(
        to: pod,
        archiveURL: archiveURL,
        sshKeyPath: "",
        workerDirectoryURL: workerDirectory
    )

    #expect(
        service.remoteJobState(for: pod, archiveURL: archiveURL)?.phase
            == .uploaded
    )
    let invocations = await executor.invocations
    #expect(invocations.count == 3)
    #expect(invocations.map(\.0.path) == [
        "/usr/bin/ssh",
        "/usr/bin/scp",
        "/usr/bin/scp",
    ])
    #expect(invocations[1].1.contains(archiveURL.path))
    #expect(
        invocations[2].1.contains(
            workerDirectory.path(percentEncoded: false)
        )
    )
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

@Test func runPodClientListOfficialPodTemplatesParsesRunPodTemplates() async throws {
    let transport = RunPodMockTransport(responses: [(
        statusCode: 200,
        data: Data("""
        [
          {
            "id": "runpod-pytorch-240",
            "name": "Runpod Pytorch 2.4.0",
            "imageName": "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
            "category": "NVIDIA",
            "isRunpod": true,
            "isServerless": false,
            "containerDiskInGb": 50,
            "volumeInGb": 20,
            "ports": ["8888/http", "22/tcp"]
          },
          {
            "id": "public-template",
            "name": "Community Template",
            "isRunpod": false,
            "isServerless": false
          },
          {
            "id": "runpod-serverless",
            "name": "Runpod Serverless Pytorch",
            "isRunpod": true,
            "isServerless": true
          }
        ]
        """.utf8)
    )])
    let client = RunPodAPIClient(
        baseURL: URL(string: "https://rest.runpod.io/v1")!,
        transport: transport
    )

    let templates = try await client.listOfficialPodTemplates(token: "rp_test_token")

    #expect(templates.map(\.id) == ["runpod-pytorch-240"])
    let template = try #require(templates.first)
    #expect(template.summaryLabel.contains("Runpod Pytorch 2.4.0"))
    #expect(template.summaryLabel.contains("runpod/pytorch"))
    #expect(template.detailLabel.contains("Official RunPod"))
    #expect(template.detailLabel.contains("container 50 GB"))

    let request = try #require(transport.requests.first)
    #expect(request.httpMethod == "GET")
    #expect(request.url?.absoluteString == "https://rest.runpod.io/v1/templates?includeRunpodTemplates=true&includePublicTemplates=false&includeEndpointBoundTemplates=false")
    #expect(request.value(forHTTPHeaderField: "Authorization") == "Bearer rp_test_token")
}

@Test func runPodTemplateRecommendationPrefersOfficialPyTorchPodTemplate() {
    let templates = [
        RunPodTemplate(
            id: "official-tensorflow",
            name: "Runpod Tensorflow",
            imageName: "runpod/tensorflow:latest",
            category: "NVIDIA",
            isRunpod: true,
            isPublic: false,
            isServerless: false,
            containerDiskInGb: 50,
            volumeInGb: 20,
            volumeMountPath: "/workspace",
            ports: ["8888/http", "22/tcp"]
        ),
        RunPodTemplate(
            id: "official-pytorch-240",
            name: "Runpod Pytorch 2.4.0",
            imageName: "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
            category: "NVIDIA",
            isRunpod: true,
            isPublic: false,
            isServerless: false,
            containerDiskInGb: 50,
            volumeInGb: 20,
            volumeMountPath: "/workspace",
            ports: ["8888/http", "22/tcp"]
        ),
        RunPodTemplate(
            id: "serverless-pytorch",
            name: "Runpod Pytorch Serverless",
            imageName: "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
            category: "NVIDIA",
            isRunpod: true,
            isPublic: false,
            isServerless: true,
            containerDiskInGb: 50,
            volumeInGb: 20,
            volumeMountPath: "/workspace",
            ports: ["8888/http"]
        )
    ]

    let recommended = RunPodTemplate.recommendedTrainingTemplate(
        for: "NVIDIA GeForce RTX 3090",
        from: templates
    )
    #expect(recommended?.id == "official-pytorch-240")

    let amdRecommended = RunPodTemplate.recommendedTrainingTemplate(
        for: "AMD Instinct MI300X OAM",
        from: templates
    )
    #expect(amdRecommended == nil)
}

@Test func mcpToolCatalogExposesRunPodLifecycleTools() throws {
    let tools = MCPToolCatalog.tools
    let names = Set(tools.compactMap { $0["name"] as? String })

    #expect(names.contains("runpod_status"))
    #expect(names.contains("refresh_runpod_launch_options"))
    #expect(names.contains("launch_runpod_pod"))
    #expect(names.contains("stop_runpod_pod"))

    let launchTool = try #require(tools.first { $0["name"] as? String == "launch_runpod_pod" })
    let schema = try #require(launchTool["inputSchema"] as? [String: Any])
    #expect(schema["required"] as? [String] == ["gpuType"])
    let properties = try #require(schema["properties"] as? [String: Any])
    #expect(properties["dryRun"] != nil)
    #expect(properties["confirm"] != nil)
    #expect(properties["useRecommendedTemplate"] != nil)
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

@Test func runPodLaunchConfigurationUsesCustomImageInsteadOfTemplateWhenBothAreFilled() throws {
    let configuration = RunPodLaunchConfiguration(
        name: "pimp-remote",
        templateID: "template-123",
        imageName: "runpod/pytorch:custom",
        gpuTypeID: "NVIDIA GeForce RTX 4090",
        interruptible: false,
        containerDiskGB: 80,
        volumeGB: 120
    )

    let requestBody = try configuration.requestBody()

    #expect(requestBody.templateId == nil)
    #expect(requestBody.imageName == "runpod/pytorch:custom")
}

@Test func runPodClientDeletePodUsesDeleteEndpoint() async throws {
    let transport = RunPodMockTransport(responses: [(
        statusCode: 200,
        data: Data()
    )])
    let client = RunPodAPIClient(
        baseURL: URL(string: "https://rest.runpod.io/v1")!,
        transport: transport
    )

    try await client.deletePod(token: "rp_test_token", podID: "abc123")

    let request = try #require(transport.requests.first)
    #expect(request.httpMethod == "DELETE")
    #expect(request.url?.absoluteString == "https://rest.runpod.io/v1/pods/abc123")
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
