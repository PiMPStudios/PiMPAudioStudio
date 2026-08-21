# Remote GPU Training

Remote GPU training lets PiMP Audio Studio prepare a reviewed training package
locally, then run heavy CUDA training on a cloud GPU such as RunPod.

The macOS app stays Swift/MLX-only. The remote worker may use Python, PyTorch,
CUDA, ACE-Step, and Side-Step because it runs on a user-approved cloud machine
outside the app.

## Current Status

Shipped today:

- Create a remote job zip from a reviewed package-v2 dataset
- Store RunPod API key in Keychain
- Configure a sandboxed copy of your SSH private key for automated SCP/SSH
- Refresh live GPU stock/pricing and official templates
- Filter GPU stock by Asia, North America, Europe, Oceania, or any region
- Automatic GPU-aware template selection (manual override available)
- Launch and terminate RunPod **Secure Cloud** pods (billable actions confirmed)
- Managed pod lifecycle: **Upload**, **Start**, **Monitor**, **Pull & Import**
- Portable `remote-worker/` install on stock RunPod PyTorch images
- Standard `remote_artifacts/**` logs, metrics, checkpoints, final adapter

Also available for power users:

- Copy upload / SSH / extract-run commands for manual Terminal flows
- Dry-run the worker before spending GPU time

## Open The Workspace

```text
Build -> Training -> Remote GPU
```

## Basic Managed Flow

1. Build and **review** a dataset package (approve selected assets).
2. Open **Training → Remote GPU**.
3. Choose adapter kind (LoRA / LoKR), recipe (Narrow / Wide), bundle
   (full mix / vocals / instrumental), base model, epochs, save interval, seed,
   and adapter name.
4. Create / export the remote job zip from the selected package.
5. Save a RunPod API key.
6. Choose / replace the SSH private key used for automated upload (stored as a
   protected sandbox copy; never placed inside the job zip).
7. Refresh GPU availability and templates; pick region + GPU type.
8. Confirm and **Launch** the pod (billable).
9. On the pod row, run left to right as controls enable:
   - **Upload** — job zip + portable worker
   - **Start** — detached remote setup/training
   - **Monitor** — reconnect status/logs (also after app relaunch)
   - **Pull & Import** — download `remote_artifacts` and import the adapter
10. **Terminate** the pod after artifacts are safe on your Mac.

Launch is Secure Cloud on-demand only, through RunPod REST API v2. If a
managed SSH private key is configured, PiMP derives its public key and merges
it into the RunPod account key set (it does not replace keys already
registered). Upload needs the pod `RUNNING` with a direct SSH endpoint
(`ssh.direct`).

### Template Selection

By default PiMP ranks official single-pod PyTorch templates for the selected
GPU. You can disable automatic mode and choose:

- An official template from the list
- A custom template ID
- A custom image name (overrides template ID)

Serverless/cluster images are rejected. Blackwell-class cards require CUDA 12.8+
and PyTorch 2.7+ when those requirements are identifiable.

## What The Zip Contains

The remote job zip contains the reviewed package plus:

```text
remote_job.json
README_REMOTE_GPU.md
run_remote_training.sh
```

The zip does **not** contain provider API keys, SSH keys, RunPod account IDs, or
Hugging Face tokens.

## Manual Terminal Flow (Optional)

If you prefer shell control:

```bash
scp -P <ssh-port> -i <key> -r remote-worker root@<pod-ip>:/workspace/pimp-remote-worker-src
ssh -p <ssh-port> -i <key> root@<pod-ip> 'bash /workspace/pimp-remote-worker-src/install.sh'
ssh -p <ssh-port> -i <key> root@<pod-ip>
pimp-remote-worker train /workspace/jobs/my-job.zip
```

Dry run:

```bash
pimp-remote-worker train /workspace/jobs/my-job.zip --dry-run
```

Doctor:

```bash
pimp-remote-worker doctor
```

Stock RunPod PyTorch images do **not** include `pimp-remote-worker` until the
portable worker is uploaded and installed.

## Output Layout

The worker writes inside the package root:

```text
remote_artifacts/
  logs.jsonl
  metrics.jsonl
  result.json
  command/
    preprocess.sh
    train.sh
    dry_run.json
  side_step_dataset.json
  tensors/
  side_step_output/
  checkpoints/
  final_adapter/
```

Import `remote_artifacts/final_adapter/` for the final adapter, or a folder under
`remote_artifacts/checkpoints/` to test a specific epoch. Pull & Import prefers
the friendly `training.adapter_name` from `result.json` when present.

## Hugging Face Tokens On The Pod

Some ACE-Step or related downloads may require a token on the pod:

```bash
export HF_TOKEN=...
export HUGGINGFACE_HUB_TOKEN=...
```

Do not put tokens inside `remote_job.json` or package zips.

## Cleanup And Billing

When the run is finished and artifacts are downloaded, **Terminate** the pod
from PiMP Audio Studio or the RunPod console.

Stopping/pausing is not enough if storage is still billing. Terminate deletes the
pod resource. Download small logs/metadata before deleting if you need support
files.

## Troubleshooting Highlights

- **Pod launched but worker missing** — run Upload / install the portable worker.
- **First Start looks hung** — initial environment setup can take 20–45 minutes
  on network storage; Monitor shows stage progress.
- **Upload disabled** — configure the SSH key and ensure a job zip is selected.
- **Pull failed but remote succeeded** — retry Pull & Import; local pullback
  errors should not destroy a finished remote job state.
- **Adapter name looks wrong** — prefer result metadata name over folder name
  `final_adapter`.

See [Troubleshooting](troubleshooting.md) and [Bug Report Checklist](bug-report-checklist.md).
