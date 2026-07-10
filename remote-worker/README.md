# PiMP Remote Worker

This folder contains the first portable PiMP Audio Studio remote worker. It is meant for stock CUDA pods such as the official RunPod PyTorch template. It is not part of the macOS app runtime.

The worker consumes an exported PiMP Audio Studio remote job zip, converts the selected `training_bundle_*.json` into Side-Step's dataset JSON format, preprocesses the audio, trains LoRA or LoKR through ACE-Step Side-Step, and writes standard `remote_artifacts/**` outputs.

## Quick RunPod Flow

From your Mac, after launching a pod and copying the job zip:

```bash
scp -P <ssh-port> -i ~/.ssh/id_ed25519 -r remote-worker root@<pod-ip>:/workspace/pimp-remote-worker-src
ssh -p <ssh-port> -i ~/.ssh/id_ed25519 root@<pod-ip> 'bash /workspace/pimp-remote-worker-src/install.sh'
```

Then on the pod:

```bash
pimp-remote-worker train /workspace/jobs/my-remote-job.zip
```

If the zip was already extracted:

```bash
cd "$(dirname "$(find /workspace/pimp-remote-jobs/current -maxdepth 4 -name remote_job.json -print -quit)")"
pimp-remote-worker train remote_job.json
```

For a cheap package/command check before starting GPU work:

```bash
pimp-remote-worker train /workspace/jobs/my-remote-job.zip --dry-run
```

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

`result.json` is the stable import point for future app-side pullback.

## Environment Variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `PIMP_REMOTE_WORKDIR` | `/workspace/pimp-remote-worker` | Persistent worker install/cache root |
| `PIMP_ACE_STEP_REPO` | `https://github.com/PiMPStudios/ACE-Step-1.5.git` | ACE-Step/Side-Step source repo |
| `PIMP_ACE_DIR` | `$PIMP_REMOTE_WORKDIR/ACE-Step-1.5` | Existing ACE-Step checkout to use |
| `PIMP_CHECKPOINT_DIR` | `$PIMP_REMOTE_WORKDIR/checkpoints` | ACE-Step model cache |
| `PIMP_REMOTE_PYTHON` | `$PIMP_REMOTE_WORKDIR/venv/bin/python` | Python executable used for training |
| `PIMP_REMOTE_DOWNLOAD_MODELS` | `1` | Download missing model assets before training |
| `PIMP_REMOTE_NUM_WORKERS` | `4` | DataLoader worker count |
| `PIMP_REMOTE_FORCE_PREPROCESS` | `0` | Recreate tensors even if cached tensors exist |
| `PIMP_REMOTE_SOURCE` | `huggingface` | ACE-Step model download source hint |

The Hugging Face token can be supplied through the normal `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN` environment variables when needed.

## Notes

- The macOS app remains Swift/MLX-only. This worker is an explicit remote boundary for CUDA/Python machines.
- Job paths must be relative to the directory containing `remote_job.json`; package escapes, symlink escapes, overlapping replaceable outputs, and unsafe zip members are rejected before training starts.
- A future public PiMP RunPod template can bake this folder into an image, but this copy/install flow keeps the first end-to-end path account-neutral.
- The worker intentionally writes the exact generated Side-Step commands before running them so failed jobs are easy to inspect.
