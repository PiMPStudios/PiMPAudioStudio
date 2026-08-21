# RunPod REST API v2 Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut PiMP Audio Studio's RunPod client over to REST API v2 and remove interruptible / spot launches, without changing the portable worker or job contract.

**Architecture:** `RunPodAPIClient` talks only to `https://api.runpod.io/v2`. Catalog GPU/stock/templates replace GraphQL. Pod SSH comes from `ssh.direct`. Launch always sends `startSsh: true` and, when a managed private key exists, GET-merges that public key into `PUT /v2/account/ssh-keys` (never replace-only).

**Tech Stack:** Swift, Swift Testing in `PiMPAudioStudioTests/RunPodServiceTests.swift`, mocked `RunPodTransport`. Verify with `xcodebuild test -scheme PiMPAudioStudio -destination 'platform=macOS' -only-testing:PiMPAudioStudioTests/RunPodServiceTests`.

**Spec:** `docs/superpowers/specs/2026-08-21-runpod-rest-v2-migration-design.md`

---

## File map

- Modify: `PiMPAudioStudio/Services/RunPodService.swift` (client, create body, pod/catalog/template decode, SSH, launch)
- Modify: `PiMPAudioStudio/UI/Training/RunPodLifecyclePanel.swift`
- Modify: `PiMPAudioStudio/UI/MainShellView.swift`
- Modify: `PiMPAudioStudio/Services/AppRuntimeIdentity.swift`
- Modify: `PiMPAudioStudio/Services/PiMPAudioCommandFacade+Training.swift`
- Modify: `PiMPAudioStudio/Services/PiMPAudioMCPToolCatalog.swift`
- Modify: `PiMPAudioStudioTests/RunPodServiceTests.swift`
- Modify: `docs/user/remote-gpu-training.md`
- Modify: `docs/remote-worker-contract.md`
- Modify: `docs/mcp-local.md` (only if it names v1 paths)
- Modify: `CHANGELOG.md`

Do not add a new Swift file unless `RunPodService.swift` becomes unworkable; `project.yml` already glob-includes `PiMPAudioStudio/`.

Default client init after the cutover:

```swift
init(
    baseURL: URL = URL(string: "https://api.runpod.io/v2")!,
    transport: any RunPodTransport = URLSessionRunPodTransport()
)
```

Remove `graphQLURL` and `gpuCatalogQuery`.

---

### Task 1: Drop interruptible

**Files:**

- Modify: `PiMPAudioStudio/Services/RunPodService.swift`
- Modify: `PiMPAudioStudio/UI/Training/RunPodLifecyclePanel.swift`
- Modify: `PiMPAudioStudio/UI/MainShellView.swift`
- Modify: `PiMPAudioStudio/Services/AppRuntimeIdentity.swift`
- Modify: `PiMPAudioStudio/Services/PiMPAudioCommandFacade+Training.swift`
- Modify: `PiMPAudioStudio/Services/PiMPAudioMCPToolCatalog.swift`
- Test: `PiMPAudioStudioTests/RunPodServiceTests.swift`

- [ ] **Step 1: Rewrite tests so interruptible is gone**

In `mcpToolCatalogExposesRunPodLifecycleTools`, assert the launch schema has no `interruptible` property.

In GPU catalog tests, replace `detailLabel(interruptible:)` with a single on-demand label. Drop the spot-price assertion.

In `RunPodLaunchConfiguration(...)` test fixtures, delete the `interruptible:` argument.

- [ ] **Step 2: Run the targeted tests and confirm they fail**

```bash
xcodebuild test -scheme PiMPAudioStudio -destination 'platform=macOS' -only-testing:PiMPAudioStudioTests/RunPodServiceTests
```

Expected: compile and/or assertion failures around `interruptible`.

- [ ] **Step 3: Remove the product surface**

Delete:

- `RunPodLifecyclePanel` `@Binding var interruptible` and the `Toggle("Interruptible", …)`
- `MainShellView` `@AppStorage(AppUserDefaultsKeys.runPodInterruptible)` and the binding / launch argument
- `AppUserDefaultsKeys.runPodInterruptible`
- MCP tool property `interruptible`
- Facade reads of `interruptible` / `spot`, payload keys `interruptible`, `spotDetail`, `supportPublicIP`

Change `RunPodLaunchConfiguration` so it no longer has `interruptible` or `supportPublicIP`.

Change `RunPodGPUType.detailLabel` to:

```swift
func detailLabel(stockStatus: RunPodStockStatus? = nil) -> String {
    var details: [String] = []
    if let memoryInGb {
        details.append("\(memoryInGb) GB VRAM")
    }
    details.append(availabilityLabel)
    if let stockStatus, stockStatus != .none {
        details.append(stockStatus.displayName)
    } else {
        details.append(stockLabel())
    }
    if let launchPriceLabel {
        details.append(launchPriceLabel)
    }
    if let maxGpuCount = maxGpuCountSecureCloud ?? maxGpuCount, maxGpuCount > 0 {
        details.append("up to \(maxGpuCount) GPU\(maxGpuCount == 1 ? "" : "s")")
    }
    return details.joined(separator: " · ")
}
```

Price helper uses only on-demand (`uninterruptablePrice` / `securePrice` / later `price.secure`). Leave `secureSpotPrice` decode unused until Task 4 drops it.

Panel GPU caption becomes `selectedRunPodGPUType.detailLabel(stockStatus:)`.

- [ ] **Step 4: Re-run Task 1 tests**

Same `xcodebuild` command. Expected: interruptible-related tests pass; remaining tests still use v1 URLs until later tasks.

- [ ] **Step 5: Commit**

```bash
git add PiMPAudioStudio/Services/RunPodService.swift \
  PiMPAudioStudio/UI/Training/RunPodLifecyclePanel.swift \
  PiMPAudioStudio/UI/MainShellView.swift \
  PiMPAudioStudio/Services/AppRuntimeIdentity.swift \
  PiMPAudioStudio/Services/PiMPAudioCommandFacade+Training.swift \
  PiMPAudioStudio/Services/PiMPAudioMCPToolCatalog.swift \
  PiMPAudioStudioTests/RunPodServiceTests.swift
git commit -m "Drop RunPod interruptible / spot launches."
```

---

### Task 2: v2 transport, wrappers, and RFC 9457 errors

**Files:**

- Modify: `PiMPAudioStudio/Services/RunPodService.swift`
- Test: `PiMPAudioStudioTests/RunPodServiceTests.swift`

- [ ] **Step 1: Point failing tests at v2**

Change every `RunPodAPIClient(baseURL: URL(string: "https://rest.runpod.io/v1")!` to `https://api.runpod.io/v2`. Remove `graphQLURL:` arguments.

Update `runPodClientInvalidAPIKeyErrorIsSpecific` body to:

```json
{
  "title": "Unauthorized",
  "status": 401,
  "detail": "JWT expired"
}
```

Still expect the existing "RunPod rejected the API key" copy, plus `JWT expired` in the detail suffix.

Delete `runPodGraphQLErrorsAreReadable` (or rewrite it in Task 4 as a catalog HTTP error). GraphQL is going away.

List-pods test fixture must wrap:

```json
{
  "pods": [
    {
      "id": "abc123",
      "name": "pytorch-training",
      "status": "RUNNING",
      "image": "runpod/pytorch:2.8",
      "cost": 0.69,
      "gpu": { "id": "NVIDIA RTX PRO 6000 Blackwell Server Edition", "count": 1 },
      "ports": ["22/tcp", "8888/http"],
      "ssh": {
        "proxy": null,
        "direct": {
          "host": "216.81.151.42",
          "port": 11646,
          "username": "root",
          "command": "ssh root@216.81.151.42 -p 11646"
        }
      }
    }
  ]
}
```

Expect URL `https://api.runpod.io/v2/pods`.

- [ ] **Step 2: Run tests — they should fail on decode / missing graphQLURL**

- [ ] **Step 3: Implement client plumbing**

```swift
struct RunPodAPIClient {
    var baseURL: URL
    var transport: any RunPodTransport

    init(
        baseURL: URL = URL(string: "https://api.runpod.io/v2")!,
        transport: any RunPodTransport = URLSessionRunPodTransport()
    ) {
        self.baseURL = baseURL
        self.transport = transport
    }
}

private struct RunPodListPodsResponse: Decodable {
    var pods: [RunPodPodSummary]
}

private struct RunPodProblemDetails: Decodable {
    var title: String?
    var status: Int?
    var detail: String?
    var errors: [String]?
}
```

`listPods` decodes `RunPodListPodsResponse` and returns `.pods`.

`validate(response:data:)` keeps status-code mapping. For the body suffix, prefer `detail` (and joined `errors`) from `RunPodProblemDetails`; otherwise the raw string as today.

Delete `makeGraphQLRequest`, `RunPodGraphQLRequest`, `RunPodGraphQLResponse`, `RunPodGraphQLError`, and `RunPodError.graphQLErrors` once nothing references them (Task 4 if catalog still needs a compile). If catalog still compiles against GraphQL in this commit, leave the GraphQL symbols until Task 4 but stop using them for pods.

`makeRequest(path:)` already appends onto `baseURL`, so path `"pods"` against `https://api.runpod.io/v2` is correct. Do not also prefix `/v2/`.

- [ ] **Step 4: Re-run tests**

List-pods URL + 401 detail should pass. Create/delete/catalog/template tests may still fail until later tasks.

- [ ] **Step 5: Commit**

```bash
git commit -m "Point the RunPod HTTP client at REST API v2."
```

---

### Task 3: Decode v2 pods and SSH direct

**Files:**

- Modify: `PiMPAudioStudio/Services/RunPodService.swift`
- Test: `PiMPAudioStudioTests/RunPodServiceTests.swift`

- [ ] **Step 1: Extend the list-pods test**

Keep the Task 2 fixture. Assert:

```swift
#expect(pods[0].statusLabel == "RUNNING")
#expect(pods[0].hourlyCostLabel == "$0.69/hr")
#expect(pods[0].sshHost == "216.81.151.42")
#expect(pods[0].sshPort == 11646)
#expect(pods[0].sshUsername == "root")
#expect(pods[0].sshCommand(sshKeyPath: "~/.ssh/id_ed25519")
    == "ssh -p 11646 -i ~/.ssh/id_ed25519 root@216.81.151.42")
```

Add a second decoded pod (or a dedicated test) with `username: "ubuntu"` and assert copy-paste / `RunPodRemoteJobPlan` use `ubuntu@`, not `root@`.

`RunPodRemoteJobPlan` currently reads `pod.publicIp` and hardcodes `root@` in scp/ssh argument arrays around `RunPodService.swift` (search `root@`). Tests in `runPodRemoteJobPlanUsesArgumentArraysAndScopedRemotePaths` must follow the new username.

- [ ] **Step 2: Run tests — fail on missing `sshHost` / decode**

- [ ] **Step 3: Decode the v2 pod shape**

`RunPodPodSummary` should decode:

- `status` → `statusLabel` (keep a private `status` or reuse `desiredStatus` as the stored field, populated from `status`)
- `cost` → hourly cost (replace `costPerHr` / `adjustedCostPerHr`)
- `gpu.id` / `gpu.count`
- `disk` → container disk
- `mounts.persistent.size/path` → volume
- `template` (string?) → `templateId`
- `ssh.direct` → host, port, username

Provide `sshHost`, `sshPort`, `sshUsername` (default `"root"` only when direct exists but username is empty). `publicIp` can remain as a computed alias of `sshHost` if MCP still emits `publicIP`.

Update `sshCommand`, `uploadCommand`, `RunPodRemoteJobPlan.init`, and the `root@\(host)` process arguments to interpolate `sshUsername`.

Coding keys must not require v1 `desiredStatus`, `publicIp`, or `portMappings`. Use `decodeIfPresent` plus nested `ssh` / `mounts` containers.

Minimal nested types:

```swift
private struct RunPodPodSSHBlock: Decodable {
    var direct: RunPodPodSSHEndpoint?
}

struct RunPodPodSSHEndpoint: Decodable, Equatable, Sendable {
    var host: String
    var port: Int
    var username: String
}
```

- [ ] **Step 4: Re-run SSH/list tests — they should pass**

- [ ] **Step 5: Commit**

```bash
git commit -m "Read RunPod v2 pod SSH from ssh.direct."
```

---

### Task 4: Replace GraphQL catalog with REST catalog

**Files:**

- Modify: `PiMPAudioStudio/Services/RunPodService.swift`
- Test: `PiMPAudioStudioTests/RunPodServiceTests.swift`

- [ ] **Step 1: Rewrite `listGPUCatalog` tests to two REST calls**

Mock transport should return, in order:

1. `GET .../gpus?include=AVAILABILITY&product=POD&cloud=SECURE` with body:

```json
{
  "gpus": [
    {
      "id": "NVIDIA GeForce RTX 4090",
      "name": "RTX 4090",
      "manufacturer": "NVIDIA",
      "memory": 24,
      "secure": true,
      "community": true,
      "price": { "secure": 0.74, "community": 0.50 },
      "maxCount": { "secure": 4, "community": 2 },
      "availability": "MEDIUM"
    },
    {
      "id": "community-only-gpu",
      "name": "Community Only",
      "manufacturer": "NVIDIA",
      "memory": 8,
      "secure": false,
      "community": true,
      "price": { "secure": 0, "community": 0.20 },
      "maxCount": { "secure": 0, "community": 1 },
      "availability": "HIGH"
    }
  ]
}
```

Second request, `GET .../datacenters?include=GPU_AVAILABILITY`, with body:

```json
{
  "dataCenters": [
    {
      "id": "US-TX-1",
      "name": "US Texas 1",
      "region": "NORTH_AMERICA",
      "gpuAvailability": [
        { "id": "NVIDIA GeForce RTX 4090", "name": "RTX 4090", "availability": "LOW" }
      ]
    },
    {
      "id": "EU-RO-1",
      "name": "EU Romania 1",
      "region": "EUROPE",
      "gpuAvailability": [
        { "id": "NVIDIA GeForce RTX 4090", "name": "RTX 4090", "availability": "HIGH" }
      ]
    }
  ]
}
```

Keep the existing region stock assertions (`northAmerica` low, `europe` high, community GPU filtered out, menu title contains `24 GB` and a price). Drop GraphQL body assertions (`lowestPrice`, `gpuTypes` query).

Assert request URLs contain `/v2/catalog/gpus` and `/v2/catalog/datacenters`.

Extend `RunPodRegion.includes` tests with a datacenter whose id does not match the old prefix but `region` is `EUROPE` (for example `IS-1` + `region: EUROPE`) so the new field is actually used.

- [ ] **Step 2: Run tests — fail (still POST graphql)**

- [ ] **Step 3: Implement REST catalog**

`listGPUCatalog`:

```swift
func listGPUCatalog(token: String) async throws -> RunPodGPUCatalog {
    async let gpuData = fetchGPUs(token: token)
    async let dcData = fetchDataCenters(token: token)
    let gpuTypes = try await gpuData
        .filter { $0.secureCloud != false }
        .sorted { $0.displayNameValue.localizedStandardCompare($1.displayNameValue) == .orderedAscending }
    let dataCenters = try await dcData.sorted {
        $0.id.localizedStandardCompare($1.id) == .orderedAscending
    }
    return RunPodGPUCatalog(gpuTypes: gpuTypes, dataCenters: dataCenters)
}
```

Map catalog GPU JSON:

- `name` → `displayName`
- `memory` → `memoryInGb`
- `secure` → `secureCloud`
- `price.secure` → `securePrice`
- `maxCount.secure` → `maxGpuCountSecureCloud`

Map datacenter GPU availability:

- `id` → `gpuTypeID`
- `name` → `displayName`
- `availability` → `stockStatus` (reuse `RunPodGPUAvailability.stockStatusValue` which already lowercases `high`/`medium`/`low`)

Add `region: String?` on `RunPodDataCenter`. Update `RunPodRegion.includes`:

```swift
func includes(_ dataCenter: RunPodDataCenter) -> Bool {
    guard self != .any else { return true }
    if let catalog = dataCenter.catalogRegion {
        return catalog == self
    }
    // existing US-/CA-/EU-/AP-/SEA-/OC- prefix fallback
}
```

Map catalog strings: `NORTH_AMERICA` → `.northAmerica`, `ASIA` → `.asia`, `EUROPE` → `.europe`, `OCEANIA` → `.oceania`. `MIDDLE_EAST` / `AFRICA` / `SOUTH_AMERICA` / `UNKNOWN` do not match those four filters (only `.any`).

Query items:

```swift
URLQueryItem(name: "include", value: "AVAILABILITY"),
URLQueryItem(name: "product", value: "POD"),
URLQueryItem(name: "cloud", value: "SECURE"),
```

and for datacenters `include=GPU_AVAILABILITY`.

Delete GraphQL types, query string, and `graphQLErrors`.

- [ ] **Step 4: Re-run catalog + region tests**

- [ ] **Step 5: Commit**

```bash
git commit -m "Load RunPod GPU stock from REST v2 catalog endpoints."
```

---

### Task 5: Official templates from the public catalog

**Files:**

- Modify: `PiMPAudioStudio/Services/RunPodService.swift`
- Test: `PiMPAudioStudioTests/RunPodServiceTests.swift`

- [ ] **Step 1: Rewrite `runPodClientListOfficialPodTemplatesParsesRunPodTemplates`**

Expect `GET https://api.runpod.io/v2/catalog/templates?source=official`.

Fixture:

```json
{
  "templates": [
    {
      "id": "runpod-pytorch-240",
      "name": "Runpod Pytorch 2.4.0",
      "image": "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
      "category": "NVIDIA",
      "public": true,
      "serverless": false,
      "disk": 50,
      "ports": ["8888/http", "22/tcp"],
      "mounts": { "persistent": { "size": 20, "path": "/workspace" } }
    },
    {
      "id": "runpod-serverless",
      "name": "Runpod Serverless Pytorch",
      "image": "runpod/pytorch:2.4.0",
      "serverless": true,
      "public": true
    }
  ]
}
```

Still expect only the official pod template, `summaryLabel` contains `runpod/pytorch`, `detailLabel` mentions container 50 GB.

`isOfficialPodTemplate` should be `serverless != true` for catalog results (all rows from `source=official` are RunPod-curated).

- [ ] **Step 2: Run test — fail (old query params / bare array)**

- [ ] **Step 3: Implement**

```swift
func listOfficialPodTemplates(token: String) async throws -> [RunPodTemplate] {
    let request = try makeRequest(
        path: "catalog/templates",
        method: "GET",
        token: token,
        queryItems: [URLQueryItem(name: "source", value: "official")]
    )
    ...
    return try JSONDecoder().decode(RunPodListTemplatesResponse.self, from: data)
        .templates
        .filter(\.isOfficialPodTemplate)
        .sorted { $0.displayNameValue.localizedStandardCompare($1.displayNameValue) == .orderedAscending }
}
```

Decode `image` into `imageName`, `serverless` into `isServerless`, `disk` into `containerDiskInGb`, `mounts.persistent` into volume fields, `public` into `isPublic`. Set `isRunpod = true` for catalog official rows (keeps `isOfficialPodTemplate` as `isRunpod == true && isServerless != true`).

Keep `recommendedTrainingTemplate` / Blackwell scoring — they already search name + image text.

- [ ] **Step 4: Re-run template tests including recommendation tests**

- [ ] **Step 5: Commit**

```bash
git commit -m "Load official RunPod templates from the v2 catalog."
```

---

### Task 6: v2 create and delete bodies

**Files:**

- Modify: `PiMPAudioStudio/Services/RunPodService.swift`
- Test: `PiMPAudioStudioTests/RunPodServiceTests.swift`

- [ ] **Step 1: Rewrite create/delete tests**

Create fixture response: HTTP 201, a v2 pod object (not wrapped), `status: PROVISIONING`, `ssh.direct: null`.

Assert URL `https://api.runpod.io/v2/pods`, method POST, Authorization bearer, body:

```swift
#expect(body["cloud"] as? String == "SECURE")
#expect(body["startSsh"] as? Bool == true)
#expect(body["templateId"] as? String == "template-123")
#expect(body["image"] == nil)
let gpu = try #require(body["gpu"] as? [String: Any])
#expect(gpu["id"] as? String == "NVIDIA GeForce RTX 4090")
#expect(gpu["count"] as? Int == 1)
#expect(body["dataCenterIds"] as? [String] == ["US-TX-1", "US-TX-3"])
#expect(body["disk"] as? Int == 80)
let mounts = try #require(body["mounts"] as? [String: Any])
let persistent = try #require(mounts["persistent"] as? [String: Any])
#expect(persistent["size"] as? Int == 120)
#expect(persistent["path"] as? String == "/workspace")
#expect(body["ports"] as? [String] == ["22/tcp", "8888/http"])
#expect(body["cloudType"] == nil)
#expect(body["gpuTypeIds"] == nil)
#expect(body["interruptible"] == nil)
#expect(body["supportPublicIp"] == nil)
```

Image-overrides-template test: body has `image` set, `templateId` omitted.

Blackwell GPU id (`NVIDIA RTX PRO 6000 Blackwell Server Edition` or `NVIDIA GeForce RTX 5090`): `gpu["minCudaVersion"] as? String == "12.8"`. Non-Blackwell omits that key.

Delete test: `DELETE https://api.runpod.io/v2/pods/abc123`. Accept 200 or 204.

- [ ] **Step 2: Run tests — fail on v1 field names**

- [ ] **Step 3: Replace `RunPodCreatePodRequest`**

```swift
struct RunPodCreatePodRequest: Encodable, Equatable, Sendable {
    var name: String
    var cloud: String
    var gpu: GPU
    var dataCenterIds: [String]?
    var image: String?
    var templateId: String?
    var disk: Int
    var ports: [String]
    var startSsh: Bool
    var mounts: Mounts

    struct GPU: Encodable, Equatable, Sendable {
        var id: String
        var count: Int
        var minCudaVersion: String?
    }

    struct Mounts: Encodable, Equatable, Sendable {
        var persistent: Persistent
    }

    struct Persistent: Encodable, Equatable, Sendable {
        var size: Int
        var path: String
    }
}
```

`RunPodLaunchConfiguration.requestBody()` fills those fields. Reuse `RunPodTemplate.isBlackwellGPU` (make it `internal`/`fileprivate` as needed) or duplicate the same `blackwell` / `geforce rtx 50` check next to the request builder.

`createPod` already uses `validate` which accepts 201.

- [ ] **Step 4: Re-run create/delete tests**

- [ ] **Step 5: Commit**

```bash
git commit -m "Create and terminate RunPod pods through REST API v2."
```

---

### Task 7: Register the managed SSH public key without wiping the account

**Files:**

- Modify: `PiMPAudioStudio/Services/RunPodService.swift`
- Test: `PiMPAudioStudioTests/RunPodServiceTests.swift`

- [ ] **Step 1: Add unit tests for key merge (pure string compare, no ssh-keygen)**

```swift
@Test func runPodSSHPublicKeyMergeAddsMissingKeyAndSkipsDuplicates() {
    let existing = [
        "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIexisting comment-one"
    ]
    let ours = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAInewkey pimp-audio-studio"
    let merged = RunPodSSHPublicKey.merge(existing: existing, inserting: ours)
    #expect(merged.count == 2)
    #expect(merged.contains(where: { $0.contains("AAAAInewkey") }))

    let already = RunPodSSHPublicKey.merge(
        existing: existing + ["ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAInewkey other-comment"],
        inserting: ours
    )
    #expect(already.count == 2)
}

@Test func runPodClientEnsureSSHKeySkipsPutWhenAlreadyRegistered() async throws {
    // GET /account/ssh-keys returns keys including ours → no PUT
}

@Test func runPodClientEnsureSSHKeyMergesPutWhenMissing() async throws {
    // GET then PUT with existing + ours; PUT body must contain both
}

@Test func runPodClientEnsureSSHKeyDoesNotPutWhenGetFails() async throws {
    // GET 401 → throws; transport.requests.filter PUT is empty
}
```

- [ ] **Step 2: Run tests — fail (types missing)**

- [ ] **Step 3: Implement**

```swift
enum RunPodSSHPublicKey {
    static func fingerprint(_ line: String) -> String {
        let parts = line.split(whereSeparator: { $0.isWhitespace })
        guard parts.count >= 2 else { return line.trimmingCharacters(in: .whitespacesAndNewlines) }
        return "\(parts[0]) \(parts[1])"
    }

    static func merge(existing: [String], inserting newKey: String) -> [String] {
        let needle = fingerprint(newKey)
        if existing.contains(where: { fingerprint($0) == needle }) {
            return existing
        }
        return existing + [newKey]
    }
}
```

Client:

```swift
func ensureAccountSSHKey(token: String, publicKey: String) async throws {
    let current = try await listAccountSSHKeys(token: token) // GET account/ssh-keys → { keys: [String] }
    let merged = RunPodSSHPublicKey.merge(existing: current, inserting: publicKey)
    guard merged != current else { return }
    try await replaceAccountSSHKeys(token: token, keys: merged) // PUT
}
```

`RunPodService.launchPod`:

1. Build request body as today.
2. If `RunPodSSHKeyStore` has a readable managed private key, derive the public key via `/usr/bin/ssh-keygen -y -f <path>` (new helper on the key store, mockable in tests through the client API with a pre-derived public key).
3. `try await client.ensureAccountSSHKey(...)` **before** `createPod`.
4. If derive or GET fails, set `lastError` and do not create.

Tests for launch-with-key can inject a `RunPodAPIClient` mock and a precomputed public key without calling `ssh-keygen` if `launchPod` accepts an optional override, **or** only unit-test `ensureAccountSSHKey` and call it from `launchPod` with a small internal helper `publicKey(fromPrivateKeyAt:)` that tests cover via a temp key generated in the test (`ssh-keygen -t ed25519 -N "" -f ...`) if the sandbox allows it. Prefer generating a throwaway key in the test process:

```bash
ssh-keygen -t ed25519 -N "" -f "$TMP/id_ed25519" -C pimp-test
```

Do not PUT when GET fails.

- [ ] **Step 4: Re-run SSH-key tests plus full `RunPodServiceTests`**

- [ ] **Step 5: Commit**

```bash
git commit -m "Register the PiMP SSH public key on RunPod without replacing existing keys."
```

---

### Task 8: Docs and changelog

**Files:**

- Modify: `docs/user/remote-gpu-training.md`
- Modify: `docs/remote-worker-contract.md`
- Modify: `PiMPAudioStudio/Services/PiMPAudioMCPToolCatalog.swift` (`stop_runpod_pod` description still says `DELETE /pods/{podId}`)
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Edit user docs**

Remove "Interruptible pricing is optional when available."

Add that launches use RunPod REST API v2, Secure Cloud on-demand only, and that PiMP registers the selected SSH public key on the RunPod account (merged with keys already there) so `startSsh` works. Upload still needs `ssh.direct` (pod `RUNNING` with port 22).

In `docs/remote-worker-contract.md` control-plane bullet, say REST API v2 rather than an unspecified REST API.

- [ ] **Step 2: Changelog under Unreleased → Changed**

- **RunPod control plane now uses REST API v2** — GPU catalog, official templates, pod list/create/terminate, and SSH registration talk to `https://api.runpod.io/v2`. GraphQL GPU listing is gone. Interruptible / spot launch is removed. Launch registers the managed SSH public key by merging it into the account key set.

- [ ] **Step 3: Run full app unit tests**

```bash
xcodebuild test -scheme PiMPAudioStudio -destination 'platform=macOS' -only-testing:PiMPAudioStudioTests
```

Expected: all `PiMPAudioStudioTests` pass, including MCP catalog tests that mention `launch_runpod_pod`.

- [ ] **Step 4: Commit**

```bash
git commit -m "Document the RunPod REST API v2 control plane."
```

---

## Live check (not automated)

After the unit tests are green, with a real key on a machine you control:

1. Refresh launch options — GPU list and official PyTorch templates populate.
2. Launch the cheapest Secure GPU in a region with stock.
3. Confirm the pod reaches `RUNNING` and `ssh.direct` is filled.
4. Upload still SCPs to `username@host -p port`.
5. Terminate via the app.
6. Confirm the RunPod console SSH key list still contains any keys that were there before, plus the PiMP key.

Do not add this as a CI test.

---

## Spec coverage

| Spec item | Task |
| --- | --- |
| Drop interruptible | 1 |
| Base URL v2, wrapped lists, RFC 9457 | 2 |
| `ssh.direct` + username | 3 |
| Catalog GPUs + datacenters, drop GraphQL | 4 |
| Official templates catalog | 5 |
| Nested create, DELETE, Blackwell CUDA | 6 |
| GET-merge-PUT SSH keys, no wipe | 7 |
| Docs / changelog | 8 |
| Worker / job zip untouched | (no task — do not edit `remote-worker/`) |
