# Remote GPU Training

Remote GPU training lets PiMP Audio Studio prepare a reviewed training package
locally, then run the heavy CUDA training work on a cloud GPU such as RunPod.

The macOS app stays Swift/MLX-only. The remote worker is allowed to use Python,
PyTorch, CUDA, ACE-Step, and Side-Step because it runs on a user-approved cloud
machine outside the app.

## Current Status

Shipped today:

- Create Remote Job zip from a reviewed dataset package.
- Store RunPod API key in Keychain.
- Refresh RunPod datacenter availability, GPU types, and official templates.
- Filter live GPU stock by Asia, North America, Europe, Oceania, or any region.
- Launch and terminate RunPod Secure Cloud pods.
- Copy upload, SSH, and extract/run commands.
- Copy the portable `remote-worker/` folder to a stock RunPod PyTorch pod.
- Run `pimp-remote-worker train <job.zip>` on the pod.
- Write remote logs, metrics, checkpoints, final adapter, and result metadata.

Still manual today:

- Uploading the zip and worker to the pod.
- Running the remote worker command.
- Downloading artifacts back to the Mac.
- Importing the adapter folder from the downloaded artifacts.

## Basic Flow

1. Build or select a reviewed dataset.
2. Open `Training -> Remote GPU`.
3. Choose adapter kind, recipe, bundle, base model, epochs, save interval, and
   seed.
4. Click `Create Remote Job`.
5. Save the zip.
6. Add or verify a RunPod API key.
7. Refresh availability and templates.
8. Select a region, then choose one of the GPU types currently in stock there.
9. Launch the pod after confirming the billable action.
10. Use the pod row buttons left to right:
    - copy upload command
    - copy SSH command
    - copy extract/run command
    - terminate pod when finished

## What The Zip Contains

The remote job zip contains the reviewed package plus:

```text
remote_job.json
README_REMOTE_GPU.md
run_remote_training.sh
```

The zip does not contain provider API keys, SSH keys, RunPod account IDs, or
Hugging Face tokens.

## Upload And Run

The app's upload command copies both:

```text
remote-worker/
your-remote-job.zip
```

On the pod, the generated run script installs the uploaded worker when needed
and then runs the job.

Manual equivalent:

```bash
scp -P <ssh-port> -i ~/.ssh/id_ed25519 -r remote-worker root@<pod-ip>:/workspace/pimp-remote-worker-src
ssh -p <ssh-port> -i ~/.ssh/id_ed25519 root@<pod-ip> 'bash /workspace/pimp-remote-worker-src/install.sh'
ssh -p <ssh-port> -i ~/.ssh/id_ed25519 root@<pod-ip>
pimp-remote-worker train /workspace/jobs/my-job.zip
```

Dry run before spending GPU time:

```bash
pimp-remote-worker train /workspace/jobs/my-job.zip --dry-run
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

Import `remote_artifacts/final_adapter/` for the final adapter, or import a
folder under `remote_artifacts/checkpoints/` to test a specific epoch.

## Hugging Face Tokens

Some ACE-Step or related model downloads may require a Hugging Face token or
accepted model license.

On the pod, provide the token through either environment variable:

```bash
export HF_TOKEN=...
export HUGGINGFACE_HUB_TOKEN=...
```

Do not put tokens inside `remote_job.json` or package zips.

## Cleanup

When the run is finished and artifacts are downloaded, terminate the pod from
the app or the RunPod console.

Stopping/pausing is not enough if storage is still billing. PiMP Audio Studio's
terminate action deletes the pod resource.
