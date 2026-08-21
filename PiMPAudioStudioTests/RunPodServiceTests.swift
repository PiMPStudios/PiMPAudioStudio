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

private func bashSyntaxStatus(for script: String) throws -> Int32 {
    let process = Process()
    let input = Pipe()
    process.executableURL = URL(fileURLWithPath: "/bin/bash")
    process.arguments = ["-n"]
    process.standardInput = input
    try process.run()
    input.fileHandleForWriting.write(Data(script.utf8))
    try input.fileHandleForWriting.close()
    process.waitUntilExit()
    return process.terminationStatus
}

@Test func apiKeyStoreKeepsRunPodOutOfLyricProviderList() {
    #expect(APIKeyStore.apiKeyProviders.contains { $0.id == APIKeyStore.runPodProviderID })
    #expect(!APIKeyStore.providers.contains { $0.id == APIKeyStore.runPodProviderID })
}

@Test func runPodClientListPodsParsesPodStatusAndSSHDetails() async throws {
    let transport = RunPodMockTransport(responses: [(
        statusCode: 200,
        data: Data("""
        {
          "pods": [
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
        }
        """.utf8)
    )])
    let client = RunPodAPIClient(
        baseURL: URL(string: "https://api.runpod.io/v2")!,
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
    #expect(transport.requests.first?.url?.absoluteString == "https://api.runpod.io/v2/pods")
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
    let sshDirectory = root.appendingPathComponent(
        "managed/.ssh",
        isDirectory: true
    )
    let sshKeyURL = sshDirectory.appendingPathComponent("id_ed25519")
    try FileManager.default.createDirectory(
        at: sshDirectory,
        withIntermediateDirectories: true
    )
    try Data("zip".utf8).write(to: archiveURL)
    try Data("#!/bin/bash".utf8).write(
        to: workerDirectory.appendingPathComponent("install.sh")
    )
    try Data("test key".utf8).write(to: sshKeyURL)

    let plan = try RunPodRemoteJobPlan(
        pod: runPodTestPod(),
        archiveURL: archiveURL,
        sshKeyPath: sshKeyURL.path,
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
        plan.startCommand.arguments.last?
            .contains("setup.in-progress") == true
    )
    #expect(
        plan.startCommand.arguments.last?
            .contains("setup.started") == true
    )
    #expect(
        plan.startCommand.arguments.last?
            .contains("nohup env") == true
    )
    #expect(
        plan.startCommand.arguments.last?
            .contains("launcher.pid") == true
    )
    #expect(
        plan.statusCommand.arguments.last?
            .contains("launcher.failed") == true
    )
    #expect(
        plan.statusCommand.arguments.last?
            .contains("estimated first setup time: 20\u{2013}45 min") == true
    )
    #expect(
        plan.preparePullbackCommand.arguments.last?
            .contains("remote_artifacts/result.json") == true
    )
    #expect(
        plan.preparePullbackCommand.arguments.last?
            .contains("-path \"$ROOT/pullback\" -prune") == true
    )
    #expect(
        plan.preparePullbackCommand.arguments.last?
            .contains("\"$ROOT/pullback.next/remote_artifacts\"") == true
    )
    #expect(
        plan.statusCommand.arguments.last?
            .contains("-path \"$ROOT/pullback\" -prune") == true
    )
    #expect(
        plan.statusCommand.arguments.contains(
            "UserKnownHostsFile=\(sshDirectory.appendingPathComponent("known_hosts-pod-test-123").path)"
        )
    )
    #expect(
        !plan.startCommand.arguments
            .joined(separator: " ")
            .localizedCaseInsensitiveContains("api_key")
    )
    #expect(
        try bashSyntaxStatus(
            for: #require(plan.startCommand.arguments.last)
        ) == 0
    )
    #expect(
        try bashSyntaxStatus(
            for: #require(plan.statusCommand.arguments.last)
        ) == 0
    )
    #expect(
        try bashSyntaxStatus(
            for: #require(plan.preparePullbackCommand.arguments.last)
        ) == 0
    )
}

@Test func runPodSSHKeyStoreImportsPrivateKeyWithRestrictedPermissions() throws {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString, isDirectory: true)
    let sourceURL = root.appendingPathComponent("source-key")
    let managedDirectoryURL = root.appendingPathComponent(
        "managed/.ssh",
        isDirectory: true
    )
    defer { try? FileManager.default.removeItem(at: root) }

    try FileManager.default.createDirectory(
        at: root,
        withIntermediateDirectories: true
    )
    let privateKey = """
    -----BEGIN OPENSSH PRIVATE KEY-----
    test-private-key
    -----END OPENSSH PRIVATE KEY-----
    """
    try Data(privateKey.utf8).write(to: sourceURL)

    let store = RunPodSSHKeyStore(
        managedDirectoryURL: managedDirectoryURL
    )
    let importedURL = try store.importPrivateKey(from: sourceURL)

    #expect(importedURL == managedDirectoryURL.appendingPathComponent("id_ed25519"))
    #expect(try String(contentsOf: importedURL, encoding: .utf8) == privateKey)
    #expect(store.hasManagedPrivateKey)

    let keyAttributes = try FileManager.default.attributesOfItem(
        atPath: importedURL.path
    )
    let directoryAttributes = try FileManager.default.attributesOfItem(
        atPath: managedDirectoryURL.path
    )
    #expect((keyAttributes[.posixPermissions] as? NSNumber)?.intValue == 0o600)
    #expect((directoryAttributes[.posixPermissions] as? NSNumber)?.intValue == 0o700)
}

@Test func runPodPullbackDirectoryUsesExplicitlyGrantedParent() throws {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString, isDirectory: true)
    let archiveDirectory = root.appendingPathComponent(
        "archives",
        isDirectory: true
    )
    let destinationParent = root.appendingPathComponent(
        "chosen destination",
        isDirectory: true
    )
    let workerDirectory = root.appendingPathComponent(
        "remote-worker",
        isDirectory: true
    )
    defer { try? FileManager.default.removeItem(at: root) }

    try FileManager.default.createDirectory(
        at: archiveDirectory,
        withIntermediateDirectories: true
    )
    try FileManager.default.createDirectory(
        at: destinationParent,
        withIntermediateDirectories: true
    )
    try FileManager.default.createDirectory(
        at: workerDirectory,
        withIntermediateDirectories: true
    )
    let archiveURL = archiveDirectory.appendingPathComponent("job.zip")
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

    let pullbackDirectory = try plan.makeLocalPullbackDirectory(
        in: destinationParent
    )

    #expect(
        pullbackDirectory.deletingLastPathComponent()
            == destinationParent.appendingPathComponent(
                "Remote Training Results",
                isDirectory: true
            )
    )
    #expect(
        FileManager.default.fileExists(
            atPath: pullbackDirectory.path(percentEncoded: false)
        )
    )
}

@Test func runPodSSHKeyStoreRejectsPublicKey() throws {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString, isDirectory: true)
    let sourceURL = root.appendingPathComponent("id_ed25519.pub")
    defer { try? FileManager.default.removeItem(at: root) }

    try FileManager.default.createDirectory(
        at: root,
        withIntermediateDirectories: true
    )
    try Data("ssh-ed25519 AAAAC3Nza test@example".utf8).write(to: sourceURL)

    let store = RunPodSSHKeyStore(
        managedDirectoryURL: root.appendingPathComponent("managed")
    )

    #expect(throws: RunPodSSHKeyStoreError.self) {
        try store.importPrivateKey(from: sourceURL)
    }
    #expect(!store.hasManagedPrivateKey)
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
          "adapter_name": "Midnight Soul LoRA",
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
        #expect(snapshot.result?.adapterName == "Midnight Soul LoRA")
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

@Test func runPodRemoteStatusPreservesPendingSetupEstimate() throws {
    let snapshot = try RunPodRemoteStatusSnapshot.parse(
        """
        PIMP_STATUS:PENDING
        Installing PyTorch, CUDA, and training dependencies. Elapsed: 9 min; estimated first setup time: 20\u{2013}45 min.
        """
    )

    if case .pending = snapshot.status {
        #expect(snapshot.result == nil)
        #expect(snapshot.logTail.contains("Elapsed: 9 min"))
        #expect(snapshot.logTail.contains("20\u{2013}45 min"))
    } else {
        Issue.record("Expected a pending remote setup snapshot")
    }
}

@Test func runPodRemoteMonitorCanReconnectWithoutLocalState() {
    #expect(
        RunPodRemoteJobControlAvailability.canMonitor(
            phase: nil,
            hasArchive: true,
            hasSSHKey: true,
            hasSSHEndpoint: true
        )
    )
    #expect(
        !RunPodRemoteJobControlAvailability.canMonitor(
            phase: nil,
            hasArchive: false,
            hasSSHKey: true,
            hasSSHEndpoint: true
        )
    )
    #expect(
        !RunPodRemoteJobControlAvailability.canMonitor(
            phase: nil,
            hasArchive: true,
            hasSSHKey: false,
            hasSSHEndpoint: true
        )
    )
    #expect(
        !RunPodRemoteJobControlAvailability.canMonitor(
            phase: .uploaded,
            hasArchive: true,
            hasSSHKey: true,
            hasSSHEndpoint: true
        )
    )
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
            ],
            "dataCenters": [
              {
                "id": "US-TX-1",
                "name": "US-TX-1",
                "location": "United States",
                "gpuAvailability": [
                  {
                    "gpuTypeId": "NVIDIA GeForce RTX 4090",
                    "displayName": "RTX 4090",
                    "stockStatus": "Low"
                  },
                  {
                    "gpuTypeId": "NVIDIA H100 PCIe",
                    "displayName": "H100 PCIe",
                    "stockStatus": ""
                  }
                ]
              },
              {
                "id": "EU-RO-1",
                "name": "EU-RO-1",
                "location": "Europe",
                "gpuAvailability": [
                  {
                    "gpuTypeId": "NVIDIA GeForce RTX 4090",
                    "displayName": "RTX 4090",
                    "stockStatus": "High"
                  }
                ]
              }
            ]
          }
        }
        """.utf8)
    )])
    let client = RunPodAPIClient(
        baseURL: URL(string: "https://api.runpod.io/v2")!,
        graphQLURL: URL(string: "https://api.runpod.io/graphql")!,
        transport: transport
    )

    let catalog = try await client.listGPUCatalog(token: "rp_test_token")
    let gpuTypes = catalog.gpuTypes

    #expect(gpuTypes.count == 2)
    #expect(catalog.dataCenters.map(\.id) == ["EU-RO-1", "US-TX-1"])
    #expect(catalog.availableGPUTypes(in: .northAmerica).map(\.id) == [
        "NVIDIA GeForce RTX 4090"
    ])
    #expect(catalog.stockStatus(for: "NVIDIA GeForce RTX 4090", in: .northAmerica) == .low)
    #expect(catalog.stockStatus(for: "NVIDIA GeForce RTX 4090", in: .europe) == .high)
    #expect(
        catalog.launchDataCenterIDs(
            for: "NVIDIA GeForce RTX 4090",
            in: .northAmerica
        ) == ["US-TX-1"]
    )
    #expect(
        catalog.launchDataCenterIDs(
            for: "NVIDIA GeForce RTX 4090",
            in: .any
        ).isEmpty
    )
    #expect(!gpuTypes.contains { $0.id == "community-only-gpu" })
    let rtx4090 = try #require(gpuTypes.first { $0.id == "NVIDIA GeForce RTX 4090" })
    #expect(rtx4090.selectionValue == "NVIDIA GeForce RTX 4090")
    #expect(rtx4090.menuTitle.contains("24 GB"))
    #expect(rtx4090.menuTitle.contains("Medium stock"))
    #expect(rtx4090.menuTitle.contains("from $0.74/hr"))
    let secureDetail = rtx4090.detailLabel()
    #expect(secureDetail.contains("Secure Cloud"))
    #expect(secureDetail.contains("Medium stock"))
    #expect(secureDetail.contains("counts 1/2"))
    #expect(secureDetail.contains("$0.74/hr"))
    #expect(!secureDetail.contains("On-demand"))
    #expect(!secureDetail.contains("Spot"))
    #expect(secureDetail.contains("up to 4 GPUs"))

    let request = try #require(transport.requests.first)
    #expect(request.httpMethod == "POST")
    #expect(request.url?.absoluteString == "https://api.runpod.io/graphql")
    #expect(request.value(forHTTPHeaderField: "Authorization") == "Bearer rp_test_token")
    let bodyData = try #require(request.httpBody)
    let body = String(data: bodyData, encoding: .utf8) ?? ""
    #expect(body.contains("gpuTypes"))
    #expect(body.contains("dataCenters"))
    #expect(body.contains("gpuAvailability"))
    #expect(body.contains("lowestPrice"))
    #expect(body.contains("stockStatus"))
    #expect(!body.contains("rp_test_token"))
}

@Test func runPodRegionsCoverRunPodsCurrentDatacenterPrefixes() {
    func dataCenter(_ id: String) -> RunPodDataCenter {
        RunPodDataCenter(
            id: id,
            name: id,
            location: "",
            gpuAvailability: []
        )
    }

    #expect(RunPodRegion.asia.includes(dataCenter("AP-JP-1")))
    #expect(RunPodRegion.asia.includes(dataCenter("SEA-SG-1")))
    #expect(RunPodRegion.northAmerica.includes(dataCenter("CA-MTL-1")))
    #expect(RunPodRegion.northAmerica.includes(dataCenter("US-TX-1")))
    #expect(RunPodRegion.europe.includes(dataCenter("EU-RO-1")))
    #expect(RunPodRegion.europe.includes(dataCenter("EUR-IS-1")))
    #expect(RunPodRegion.oceania.includes(dataCenter("OC-AU-1")))
    #expect(!RunPodRegion.europe.includes(dataCenter("US-TX-1")))
    #expect(RunPodRegion.any.includes(dataCenter("US-TX-1")))
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
        baseURL: URL(string: "https://api.runpod.io/v2")!,
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
    #expect(request.url?.absoluteString == "https://api.runpod.io/v2/templates?includeRunpodTemplates=true&includePublicTemplates=false&includeEndpointBoundTemplates=false")
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

@Test func runPodTemplateRecommendationRequiresBlackwellCompatibleRuntime() {
    let legacyTemplate = RunPodTemplate(
        id: "runpod-torch-v240",
        name: "Runpod Pytorch 2.4.0",
        imageName: "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
        category: "NVIDIA",
        isRunpod: true,
        isPublic: true,
        isServerless: false,
        containerDiskInGb: 20,
        volumeInGb: 20,
        volumeMountPath: "/workspace",
        ports: ["8888/http", "22/tcp"]
    )
    let blackwellTemplate = RunPodTemplate(
        id: "runpod-torch-v280",
        name: "Runpod Pytorch 2.8.0",
        imageName: "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",
        category: "NVIDIA",
        isRunpod: true,
        isPublic: true,
        isServerless: false,
        containerDiskInGb: 30,
        volumeInGb: 50,
        volumeMountPath: "/workspace",
        ports: ["8888/http", "22/tcp"]
    )
    let clusterTemplate = RunPodTemplate(
        id: "runpod-torch-v291-cluster",
        name: "Runpod Pytorch 2.9.1 for clusters",
        imageName: "runpod/pytorch:1.0.7-cu1300-torch291-ubuntu2404-cluster",
        category: "NVIDIA",
        isRunpod: true,
        isPublic: true,
        isServerless: false,
        containerDiskInGb: 50,
        volumeInGb: 30,
        volumeMountPath: "/workspace",
        ports: ["8888/http", "22/tcp"]
    )

    let templates = [legacyTemplate, clusterTemplate, blackwellTemplate]
    let recommended = RunPodTemplate.recommendedTrainingTemplate(
        for: "NVIDIA RTX PRO 4500 Blackwell",
        from: templates
    )
    #expect(recommended?.id == "runpod-torch-v280")

    let rtx5090Recommended = RunPodTemplate.recommendedTrainingTemplate(
        for: "NVIDIA GeForce RTX 5090",
        from: templates
    )
    #expect(rtx5090Recommended?.id == "runpod-torch-v280")

    let legacyOnly = RunPodTemplate.recommendedTrainingTemplate(
        for: "NVIDIA RTX PRO 4500 Blackwell",
        from: [legacyTemplate]
    )
    #expect(legacyOnly == nil)
    #expect(
        legacyTemplate.incompatibilityReason(
            for: "NVIDIA RTX PRO 4500 Blackwell"
        )?.contains("CUDA 12.8") == true
    )
    #expect(
        clusterTemplate.incompatibilityReason(
            for: "NVIDIA RTX PRO 4500 Blackwell"
        )?.contains("Cluster") == true
    )
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
    #expect(properties["interruptible"] == nil)
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
        baseURL: URL(string: "https://api.runpod.io/v2")!,
        transport: transport
    )
    let configuration = RunPodLaunchConfiguration(
        name: "pimp-remote",
        templateID: "template-123",
        imageName: "",
        gpuTypeID: "NVIDIA GeForce RTX 4090",
        dataCenterIDs: ["US-TX-3", "US-TX-1", "US-TX-3"],
        containerDiskGB: 80,
        volumeGB: 120
    )

    let pod = try await client.createPod(token: "secret-token", configuration: configuration)

    #expect(pod.id == "newpod")
    let request = try #require(transport.requests.first)
    #expect(request.httpMethod == "POST")
    #expect(request.url?.absoluteString == "https://api.runpod.io/v2/pods")
    #expect(request.value(forHTTPHeaderField: "Authorization") == "Bearer secret-token")
    #expect(request.value(forHTTPHeaderField: "Content-Type") == "application/json")

    let bodyData = try #require(request.httpBody)
    let body = try #require(JSONSerialization.jsonObject(with: bodyData) as? [String: Any])
    #expect(body["templateId"] as? String == "template-123")
    #expect(body["computeType"] as? String == "GPU")
    #expect(body["cloudType"] as? String == "SECURE")
    #expect(body["gpuTypeIds"] as? [String] == ["NVIDIA GeForce RTX 4090"])
    #expect(body["dataCenterIds"] as? [String] == ["US-TX-1", "US-TX-3"])
    #expect(body["dataCenterPriority"] as? String == "availability")
    #expect(body["volumeInGb"] as? Int == 120)
    #expect(!String(data: bodyData, encoding: .utf8)!.contains("secret-token"))
}

@Test func runPodLaunchConfigurationUsesCustomImageInsteadOfTemplateWhenBothAreFilled() throws {
    let configuration = RunPodLaunchConfiguration(
        name: "pimp-remote",
        templateID: "template-123",
        imageName: "runpod/pytorch:custom",
        gpuTypeID: "NVIDIA GeForce RTX 4090",
        containerDiskGB: 80,
        volumeGB: 120
    )

    let requestBody = try configuration.requestBody()

    #expect(requestBody.templateId == nil)
    #expect(requestBody.imageName == "runpod/pytorch:custom")
    #expect(requestBody.dataCenterIds == nil)
    #expect(requestBody.dataCenterPriority == nil)
}

@Test func runPodClientDeletePodUsesDeleteEndpoint() async throws {
    let transport = RunPodMockTransport(responses: [(
        statusCode: 200,
        data: Data()
    )])
    let client = RunPodAPIClient(
        baseURL: URL(string: "https://api.runpod.io/v2")!,
        transport: transport
    )

    try await client.deletePod(token: "rp_test_token", podID: "abc123")

    let request = try #require(transport.requests.first)
    #expect(request.httpMethod == "DELETE")
    #expect(request.url?.absoluteString == "https://api.runpod.io/v2/pods/abc123")
    #expect(request.value(forHTTPHeaderField: "Authorization") == "Bearer rp_test_token")
}

@Test func runPodClientInvalidAPIKeyErrorIsSpecific() async throws {
    let transport = RunPodMockTransport(responses: [(
        statusCode: 401,
        data: Data(#"{"title":"Unauthorized","status":401,"detail":"JWT expired"}"#.utf8)
    )])
    let client = RunPodAPIClient(
        baseURL: URL(string: "https://api.runpod.io/v2")!,
        transport: transport
    )

    do {
        _ = try await client.listPods(token: "expired-token")
        Issue.record("Expected invalid API key error")
    } catch {
        #expect(error.localizedDescription.contains("RunPod rejected the API key"))
        #expect(error.localizedDescription.contains("generate a new key"))
        #expect(error.localizedDescription.contains("JWT expired"))
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
