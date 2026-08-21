# RunPod REST API v2 Migration — Design Spec

## Goal

Move PiMP Audio Studio's RunPod control plane off REST v1
(`https://rest.runpod.io/v1`) and GraphQL (`https://api.runpod.io/graphql`) onto
REST API v2 (`https://api.runpod.io/v2`) before v1 retires on 15 November 2026.
Drop interruptible / spot launches. Keep the managed Upload / Start / Monitor /
Pull flow, Keychain API keys, sandboxed SSH private key, MCP tool names, and
portable `remote-worker/` unchanged.

## Why Interruptible Goes Away

Interruptible (spot) Secure Cloud is cheaper because the provider can reclaim
the GPU. Remote LoRA / LoKR training is a long install + preprocess + train
job on host-local `/workspace` storage, not a network volume. A reclaim:

- kills the machine and the in-flight run
- still bills for time already used
- can leave a half-written adapter even when the worker supports checkpoints
- is not documented on the v2 create-pod body

PiMP has never needed this for the product. On-demand Secure Cloud only.

## Out Of Scope

- `remote-worker/` Python / Side-Step / CUDA install
- Job zip and `remote_artifacts/**` contract
- MCP tool names (`launch_runpod_pod`, `stop_runpod_pod`, and the managed job
  tools)
- Community Cloud, serverless, clusters, billing APIs, pod log streaming
- Dual-stack v1/v2 runtime flag
- Splitting `RunPodService.swift` unless the file change forces it

## Architecture

`RunPodAPIClient` remains the only HTTP client. `RunPodService` still owns
UI/MCP state. Bearer API keys stay as they are.

Default base URL becomes `https://api.runpod.io/v2`. The GraphQL URL and
`gpuCatalogQuery` go away.

v2 list responses are wrapped (`{"pods":[...]}`, `{"templates":[...]}`,
`{"gpus":[...]}`, `{"dataCenters":[...]}`). Create still returns a single pod
object (HTTP 201). Terminate stays `DELETE /pods/{id}`.

## Endpoint Map

| Current | v2 |
| --- | --- |
| `GET /v1/pods` bare array | `GET /v2/pods` → `{ "pods": [...] }` |
| `POST /v1/pods` flat body | `POST /v2/pods` nested body |
| `DELETE /v1/pods/{id}` | `DELETE /v2/pods/{id}` |
| `GET /v1/templates?includeRunpodTemplates=true…` | `GET /v2/catalog/templates?source=official` |
| GraphQL `gpuTypes` + `dataCenters` | `GET /v2/catalog/gpus?include=AVAILABILITY&product=POD&cloud=SECURE` and `GET /v2/catalog/datacenters?include=GPU_AVAILABILITY` |
| (none) | `GET /v2/account/ssh-keys` then merge `PUT /v2/account/ssh-keys` when launching with a managed private key |

`GET /v2/templates` is the caller's own templates only. Official PyTorch images
must come from the catalog.

## Create Body

Always Secure Cloud, one GPU, SSH + TCP 22, host-local `/workspace`.

```json
{
  "name": "PiMP Audio Studio Remote Job",
  "cloud": "SECURE",
  "gpu": {
    "id": "NVIDIA GeForce RTX 4090",
    "count": 1
  },
  "dataCenterIds": ["US-TX-3"],
  "templateId": "30zmvf89kd",
  "disk": 80,
  "ports": ["22/tcp", "8888/http"],
  "startSsh": true,
  "mounts": {
    "persistent": {
      "size": 120,
      "path": "/workspace"
    }
  }
}
```

Rules:

- `startSsh` is always `true`. v2 defaults it to false; without it Upload/SCP
  cannot work.
- Custom image still overrides template ID (send `image`, omit `templateId`).
- `dataCenterIds` omitted when the region is "any" and the catalog did not
  narrow placement.
- Blackwell-class GPU types also send `gpu.minCudaVersion`: `"12.8"` (same
  rule as today's template ranking).
- Do not send `interruptible`, `supportPublicIp`, `cloudType`, `computeType`,
  `gpuTypeIds`, `gpuTypePriority`, `dataCenterPriority`, `imageName`,
  `containerDiskInGb`, `volumeInGb`, or `volumeMountPath`.

## Pod SSH

v2 does not give `publicIp` + `portMappings["22"]`. Use `ssh.direct`:

- host, port, username (usually `root`)
- `ssh.direct` is null until the pod is `RUNNING` with `22/tcp` published

`RunPodPodSummary` keeps `publicIp` / `sshPort` as computed views of
`ssh.direct` so copy-paste commands and `RunPodRemoteJobPlan` stay familiar.
Username must come from `ssh.direct.username` instead of a hardcoded `root@`.
Do not use `ssh.proxy` for SCP (proxy is interactive shell only).

Status comes from `status` (`PROVISIONING`, `STARTING`, `RUNNING`, `EXITED`,
`ERROR`, `TERMINATED`), mapped into the existing `statusLabel`. Hourly cost
comes from `cost`. GPU label comes from `gpu.id`. Disk / volume come from
`disk` and `mounts.persistent`.

## Catalog Mapping

GPU types:

- `id` → selection id (unchanged)
- `name` → display name
- `memory` → VRAM GB
- `secure` → Secure Cloud flag; drop types with `secure == false`
- `price.secure` → on-demand hourly
- `maxCount.secure` → GPU ceiling
- `availability` plus per-datacenter `gpuAvailability[].availability` →
  `NONE|LOW|MEDIUM|HIGH` (no `availableGpuCounts`)

Data centers:

- Prefer catalog `region` (`NORTH_AMERICA`, `EUROPE`, `ASIA`, `OCEANIA`, …)
  when matching `RunPodRegion`
- Keep the current ID-prefix fallback (`US-`/`CA-`, `EU-`/`EUR-`, `AP-`/`SEA-`,
  `OC-`) when `region` is missing or `UNKNOWN`

Official templates:

- `image` → image name
- `serverless` true → reject as today
- `disk` / `mounts.persistent` → container and volume defaults
- Treat `source=official` and `serverless != true` as official pod templates
- Ranking (PyTorch, CUDA, Blackwell, cluster reject) stays on name + image text

## Account SSH Keys

v2 only injects keys registered on the account. `PUT /v2/account/ssh-keys`
**replaces the entire key set**. Never PUT only the PiMP key.

On launch, when a managed private key is configured:

1. Derive the public key with `/usr/bin/ssh-keygen -y -f <private>`.
2. `GET /v2/account/ssh-keys`.
3. If GET fails, abort launch (do not PUT).
4. If the key material (type + base64, ignore comment) is already present,
   skip PUT.
5. Otherwise PUT the existing keys plus the PiMP public key.

If no managed private key is configured, still send `startSsh: true` and leave
account keys alone. Upload remains gated on a readable sandbox key as today.

Passphrase-protected keys that `ssh-keygen -y` cannot read fail launch with a
clear message (unencrypted key, or register the `.pub` in the RunPod console).

## Errors

Parse RFC 9457 `{ "title", "status", "detail", "errors"? }` and keep the
existing 401 / 403 / 404 / 429 / 5xx user-facing copy, using `detail` when
present. Fall back to raw body text if the payload is not a problem object.

## MCP And UI

Remove:

- Remote GPU **Interruptible** toggle
- `@AppStorage` `runPod.interruptible` (delete the defaults key; do not migrate
  old values)
- `launch_runpod_pod` `interruptible` property
- MCP payloads `interruptible`, `spotDetail`, `supportPublicIP`

GPU detail labels show Secure Cloud on-demand price only.

Keep `region` / `dataCenterIDs`, dry-run, confirm, recommended template, and
billable confirmation.

## Testing

Rewrite `PiMPAudioStudioTests/RunPodServiceTests.swift` against v2 URLs and
wrapped JSON. No live RunPod network tests. Cover:

- list/create/delete paths and Authorization header
- create body shape (no v1 fields, `startSsh: true`, image-overrides-template)
- GPU catalog from REST, Secure-only filter, region stock
- official templates from catalog, serverless dropped
- RFC 9457 401 detail
- SSH direct host/port/username in commands and `RunPodRemoteJobPlan`
- SSH key ensure: skip PUT when already registered; merge PUT when not;
  abort if GET fails
- MCP catalog no longer advertises `interruptible`

## Docs

Update `docs/user/remote-gpu-training.md`, `docs/remote-worker-contract.md`,
`docs/mcp-local.md` / MCP tool descriptions that mention v1 DELETE paths, and
`CHANGELOG.md`.
