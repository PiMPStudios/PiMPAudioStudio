# Remote Worker

## Role

`remote-worker/` is a portable Python package for stock CUDA pods (especially
official RunPod PyTorch templates). The macOS app uploads it as an inert payload
and never executes it locally.

It consumes a PiMP remote job zip, resolves package-v2 selections, converts them
to Side-Step dataset JSON, preprocesses tensors, trains LoRA/LoKR through
ACE-Step Side-Step, and writes `remote_artifacts/**`.

## Operator Flow

Managed from the app (preferred):

```text
Upload -> Start -> Monitor -> Pull & Import -> Terminate
```

Manual:

```bash
scp -P <port> -i <key> -r remote-worker root@<ip>:/workspace/pimp-remote-worker-src
ssh -p <port> -i <key> root@<ip> 'bash /workspace/pimp-remote-worker-src/install.sh'
pimp-remote-worker train /workspace/jobs/job.zip
pimp-remote-worker train /workspace/jobs/job.zip --dry-run
pimp-remote-worker doctor
```

## Install Notes

- Prefers a Python 3.11/3.12 interpreter that already imports the template CUDA
  Torch stack when possible.
- Verifies/installs compatible `torchaudio` and TorchCodec matched to Torch.
- Re-enters the managed venv when launched from `/usr/local/bin`.
- Pins `PIMP_REMOTE_PYTHON` so helpers see `huggingface_hub` etc.

## Environment Variables

| Variable | Default / purpose |
| --- | --- |
| `PIMP_REMOTE_WORKDIR` | `/workspace/pimp-remote-worker` install root |
| `PIMP_ACE_STEP_REPO` | ACE-Step/Side-Step source repo URL |
| `PIMP_ACE_DIR` | Existing checkout override |
| `PIMP_CHECKPOINT_DIR` | Model cache |
| `PIMP_REMOTE_PYTHON` | Python executable |
| `PIMP_REMOTE_DOWNLOAD_MODELS` | Download missing assets (default on) |
| `PIMP_REMOTE_NUM_WORKERS` | DataLoader workers |
| `PIMP_REMOTE_FORCE_PREPROCESS` | Rebuild tensors |
| `HF_TOKEN` / `HUGGINGFACE_HUB_TOKEN` | Hugging Face auth on pod |

## Safety

- Package paths must be relative to the directory containing `remote_job.json`.
- Reject absolute/parent-relative escapes and unsafe zip members (including
  symlinks) before replace.
- Destructive outputs cannot target package root or overlap staging unsafely.
- Training ranges longer than ACE-Step's 240s **per-clip** preprocess max
  (`MAX_CLIP_DURATION_SECONDS` in `pimp-remote-worker`) must be explicit
  valid ranges; shorter materialization uses `ffmpeg` when needed.
- Stem-only packages stage unique filenames to avoid `vocals.wav` collisions.
- Commands run from a common safe root (for example `/workspace`) for ACE-Step
  `safe_path` constraints.
- Treat printed `[FAIL]` / `TRAINING FAILED` as failures even if exit code is 0.
- Default precision path resolves `bf16_or_fp16` carefully (fp16 preprocess +
  bf16 train defaults) and validates tensors for NaN/Inf.

## Output Contract

```text
remote_artifacts/
  logs.jsonl
  metrics.jsonl
  result.json
  command/preprocess.sh
  command/train.sh
  command/dry_run.json
  side_step_dataset.json
  tensors/
  side_step_output/
  checkpoints/
  final_adapter/
```

`result.json` is the stable import point. Include `training.adapter_name` when
possible so Pull & Import can use a friendly local name.

## App Integration Points

- Job builder: `RemoteTrainingJob.swift`
- Control plane: `RunPodService.swift`
- UI: `RunPodLifecyclePanel.swift` + Training Remote workspace
- MCP: `runpod_status`, `refresh_runpod_launch_options`, `launch_runpod_pod`,
  `stop_runpod_pod`

## Future Work

See `docs/remote-sidestep-backend-plan.md` for Side-Step source jobs (full
sources + rich sidecars, tensor-cache reuse, recipe sweeps) beyond the current
clip/package worker.
