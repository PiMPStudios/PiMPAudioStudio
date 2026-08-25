# Storage, Sandbox, And Security

## Bundle And Identity

| Key | Value |
| --- | --- |
| Bundle ID | `com.pimpstudios.PiMPAudioStudio` |
| Display name | PiMP Audio Studio |
| Sandbox | enabled |
| Hardened runtime | enabled |
| Network client | enabled |
| Network server | enabled (localhost MCP) |

See `project.yml` and `PiMPAudioStudio.entitlements`.

## Important Paths

### Container Application Support

```text
~/Library/Containers/com.pimpstudios.PiMPAudioStudio/Data/Library/Application Support/PiMP Audio Studio/
```

Common children:

| Path | Contents |
| --- | --- |
| `models/` | Downloaded model weights + adapters subdirectory |
| `Diagnostics/` | AUDIODIAG log files |
| other app state | presets, indexes, migration markers as applicable |

### User Output Directory

Chosen at first run via `OutputDirectoryManager`. Holds History audio/traces,
`TrainingDatasets/`, exports, and related user artifacts outside (or linked from)
the container depending on security-scoped bookmarks.

### Remote Worker On Pod

```text
/workspace/pimp-remote-worker
/workspace/jobs/...
.../remote_artifacts/
```

## Entitlements Snapshot

From project configuration:

- App sandbox
- User-selected file read/write
- Network client + server
- Temporary absolute read-only exceptions for common CLI prefixes
  (`/opt/homebrew/`, `/usr/local/`, `/opt/local/`)
- Home-relative read-only exceptions for Hugging Face cache, Codex, Grok
- Home-relative read/write exception for Gemini state

Review temporary exceptions before public paid release. Prefer security-scoped
user selection when a workflow can use it.

## Secrets Handling

| Secret | Storage | Must not appear in |
| --- | --- | --- |
| Hugging Face token | Keychain | model folders, zips, traces, logs |
| Provider API keys | Keychain (optional env fallback) | packages, MCP resources |
| RunPod API key | Keychain | job zips |
| RunPod SSH private key | Protected sandbox copy (`0700`/`0600`) | job zips, logs |
| MCP bearer token | Keychain | bug reports |

## Model File Checksums

Built-in ACE-Step files pin SHA-256 values from Hugging Face. After download,
`DownloadClient.downloadAndVerify` retries once on mismatch, then deletes the
file and fails.

A missing or empty pin **skips** verification. Gated catalogs (Stable Audio)
must still download. Do not fail closed on a missing hash.

## Path Safety Rules

Implemented across History, adapters, model indexes, remote worker, and MCP:

- Reject absolute and `..` escapes for package-relative paths
- Contain history/audio file access to History directory
- Adapter registry requires matching id and contained directory
- Zip extraction rejects traversal and symlink members
- Corrupt indexes (`history.json`, `user-models.json`) are preserved as
  `.corrupt` rather than silently destroying recoverability during sweeps

## MCP Network Exposure

MCP listens on loopback only. Still:

- Keep the server off by default
- Prefer token mode on shared Macs
- Do not port-forward the MCP port to untrusted networks

## Billable Cloud Guardrails

- RunPod pod launch requires explicit UI confirmation or MCP confirm/elicitation
- Defaults lean dry-run for MCP launch tools
- Terminate uses DELETE (destroy) not pause-with-storage-billing
- UI warns about data loss on terminate

## Checklist For New Features Touching Files Or Network

1. Does it need a new entitlement? Prefer user-selected URLs.
2. Are writes confined to package/output/container roots?
3. Could a malicious filename escape storage?
4. Could logs leak tokens or private audio paths?
5. Are destructive/billable actions confirmation-gated?
