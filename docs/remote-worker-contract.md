# Remote Worker Contract

This document defines the provider-neutral remote job shape for PiMP Audio Studio dataset annotation, preprocessing, LoRA training, and LoKR training.

The contract is intentionally separate from RunPod, DigitalOcean, or any other provider. Provider adapters only upload packages, start workers, stream status/logs, cancel jobs, and download artifacts. The package, job request, logs, metrics, and results should look the same no matter where the GPU runs.

## Goals

- Keep PiMP Audio Studio's local dataset package as the source of truth.
- Let remote workers consume the same zip that local training can re-import.
- Avoid provider-specific fields in dataset packages and training manifests.
- Keep provider API keys in Keychain, never inside package zips or job files.
- Make logs, metrics, checkpoints, final adapters, and updated package metadata importable by the app.

## Non-Goals

- No direct provider API integration in this phase.
- No automatic audio upload without explicit user action.
- No provider-specific worker image or billing policy in package metadata.
- No Python dependency inside the macOS app. Remote workers may use whatever runtime they need behind an explicit remote boundary.

## Package Root

An exported package zip keeps the dataset package directory intact. The worker should unzip it and treat the package root as the working directory.

Required or expected package files:

```text
remote_training_package.json
dataset_manifest.json
captions.json
training_bundle_full_mix.json
training_bundle_vocals.json
training_bundle_instrumental.json
preprocess_cache.json
annotations/**
preprocessed/**
stems/**
```

Stem bundles are optional until local enrichment has run. `preprocessed/**` can be empty, partially filled, or complete.

## Job Request

Before launch, PiMP Audio Studio creates a provider-neutral `remote_job.json`. The provider adapter sends that file with the package zip or writes it into the unzipped package before the worker starts.

The Training workspace's **Remote GPU -> Create Remote Job** export writes `remote_job.json`, `README_REMOTE_GPU.md`, and `run_remote_training.sh` into the package root. The shell script is a thin handoff for an installed `pimp-remote-worker` command or official provider template; provider credentials still stay outside the archive.

Minimal shape:

```json
{
  "pimp_audio_remote_job_version": "1.0",
  "job_id": "uuid",
  "created_at": "2026-05-29T19:00:00Z",
  "task": "train_lora",
  "package": {
    "package_name": "TheDarkGentlemen_local_2026-05-29-120000",
    "package_root": ".",
    "descriptor": "remote_training_package.json",
    "entry_bundle": "training_bundle_full_mix.json",
    "selected_bundle": {
      "name": "full_mix",
      "path": "training_bundle_full_mix.json",
      "include": ["full_mix"],
      "training_lyrics": "[Instrumental]",
      "training_lyrics_source": "constant"
    },
    "preprocess_cache": "preprocess_cache.json",
    "copy_rules": {
      "include": ["clips/**", "stems/**", "annotations/**", "preprocessed/**"],
      "exclude": []
    }
  },
  "training": {
    "adapter_mode": "lora",
    "adapter_name": "TheDarkGentlemen-lora",
    "base_model_id": "ace-step-xl-sft",
    "base_model_label": "ACE-Step XL SFT",
    "target_layers": ["self_attn", "cross_attn"],
    "training_mode": "corrected"
  },
  "runtime": {
    "gpu_class": "h100_or_h200",
    "precision": "bf16_or_fp16"
  },
  "outputs": {
    "logs_file": "remote_artifacts/logs.jsonl",
    "metrics_file": "remote_artifacts/metrics.jsonl",
    "result_file": "remote_artifacts/result.json",
    "checkpoints_directory": "remote_artifacts/checkpoints",
    "final_adapter_directory": "remote_artifacts/final_adapter",
    "updated_package_directory": "remote_artifacts/package"
  },
  "safety": {
    "upload_requires_user_approval": true,
    "redact_local_paths": true,
    "write_provider_secrets_to_package": false,
    "secrets_required": []
  }
}
```

Supported `task` values:

- `preprocess`
- `annotate`
- `train_lora`
- `train_lokr`

## Worker Responsibilities

The worker should:

1. Read `remote_job.json`.
2. Read `remote_training_package.json`.
3. Resolve paths from `package.package_root`.
4. Use `package.entry_bundle` as the selected training bundle.
5. Write append-only logs to `remote_artifacts/logs.jsonl`.
6. Write metrics events to `remote_artifacts/metrics.jsonl`.
7. Write checkpoints under `remote_artifacts/checkpoints/`.
8. Write the final adapter under `remote_artifacts/final_adapter/`.
9. Write `remote_artifacts/result.json` at the end.
10. If the worker changes package metadata, write the updated package under `remote_artifacts/package/`.

Workers may fill or reuse `preprocessed/<bundle>/samples/*.safetensors` and `preprocessed/<bundle>/index.json`. They should refresh `preprocess_cache.json` when cache status changes.

## Result File

`remote_artifacts/result.json` is the stable import point.

```json
{
  "pimp_audio_remote_result_version": "1.0",
  "job_id": "uuid",
  "task": "train_lora",
  "status": "succeeded",
  "started_at": "2026-05-29T19:01:00Z",
  "finished_at": "2026-05-29T20:10:00Z",
  "artifacts": [
    {
      "kind": "final_adapter",
      "path": "remote_artifacts/final_adapter",
      "media_type": "application/vnd.pimp-audio.adapter"
    }
  ],
  "metrics": {
    "best_loss": 0.5014
  }
}
```

Supported `status` values:

- `queued`
- `running`
- `succeeded`
- `failed`
- `cancelled`

Failures should include:

```json
{
  "error": {
    "code": "out_of_memory",
    "message": "Worker ran out of GPU memory while training epoch 3.",
    "recoverable": true
  }
}
```

## Logs And Metrics

`logs.jsonl` is for human-readable worker events:

```json
{"ts":"2026-05-29T19:01:00Z","level":"info","stage":"setup","message":"Loaded package"}
```

`metrics.jsonl` is for graphable values:

```json
{"ts":"2026-05-29T19:10:00Z","epoch":1,"step":42,"loss":0.82,"lr":0.0001}
```

The app should be able to stream or poll both files without caring which provider hosts the worker.

## Provider Adapter Boundary

Provider adapters are thin. A RunPod adapter and a DigitalOcean adapter should implement the same app-side operations:

- prepare upload
- upload package zip and `remote_job.json`
- launch worker
- poll or stream job status
- stream logs
- cancel job
- download `remote_artifacts/**`
- clean up remote resources

Provider adapters may store provider-specific IDs locally in PiMP Audio Studio, but those IDs do not belong in the dataset package or `remote_job.json`.

### RunPod Control Plane

The first RunPod integration is intentionally app-local provider control, not a package-schema change:

- RunPod API keys are stored through the same Keychain-backed `APIKeyStore` path as other remote credentials, with `RUNPOD_API_KEY` as the environment fallback.
- The Remote GPU panel can list pods, request a new pod from a template or image, and stop pods through RunPod's REST API.
- Pod launch is always behind a confirmation prompt because it can start billable GPU time.
- The app can copy SSH/upload handoff commands for the most recently exported job zip.

Worker logs, progress metrics, artifact pullback, and adapter import should still come from the provider-neutral `remote_artifacts/**` contract. RunPod pod status is a host lifecycle signal; it is not a replacement for worker-level training status.

## Security And Privacy

- Provider API tokens must be stored in Keychain.
- `remote_job.json` must not contain API keys, bearer tokens, SSH keys, or provider account IDs.
- Absolute local paths should be redacted before upload.
- Audio upload must require explicit user approval.
- Launch UI must show requested GPU class and make billable pod creation an explicit user confirmation. Future upload UI should also show estimated package size, max runtime, and optional max cost before transfer.

## First Implementation Path

1. Generate and validate `remote_job.json` locally.
2. Export package zip plus job request.
3. Add a local "dry-run worker" command or test harness that reads the zip and writes fake `remote_artifacts`.
4. Add provider settings and Keychain-backed credentials. Shipped for RunPod API tokens.
5. Implement one provider adapter. Partially shipped for RunPod pod list/launch/stop and SSH upload handoff commands.
6. Add the second provider adapter against the same app-side protocol.

Remaining RunPod work should focus on upload execution, worker log/progress polling, artifact download, and one-click adapter import without changing the provider-neutral package/result shape.
