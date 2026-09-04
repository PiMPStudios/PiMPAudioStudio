from __future__ import annotations

import argparse
import importlib.util
import json
import tempfile
import unittest
from importlib.machinery import SourceFileLoader
from io import StringIO
from pathlib import Path
from unittest.mock import patch

WORKER_PATH = Path(__file__).resolve().parents[1] / "pimp-remote-worker"
LOADER = SourceFileLoader("pimp_remote_worker_training", str(WORKER_PATH))
SPEC = importlib.util.spec_from_loader(LOADER.name, LOADER)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not load remote worker from {WORKER_PATH}")
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)


def write_training_job(root: Path, *, task: str = "train_lora", training: dict | None = None) -> Path:
    job = {
        "job_id": "training-test",
        "task": task,
        "package": {
            "package_root": ".",
            "entry_bundle": "training_bundle.json",
        },
        "training": training or {},
    }
    path = root / "remote_job.json"
    path.write_text(json.dumps(job), encoding="utf-8")
    (root / "audio.wav").write_bytes(b"fixture-audio")
    (root / "training_bundle.json").write_text(
        json.dumps(
            {
                "clips": [
                    {
                        "id": "clip-a",
                        "file": "audio.wav",
                        "caption": "NEW caption",
                        "duration": 1,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return path


def worker_args(job: Path) -> argparse.Namespace:
    return argparse.Namespace(job=str(job), dry_run=False, no_download=True)


class RemoteWorkerTrainingTests(unittest.TestCase):
    def test_preprocess_only_job_does_not_start_training(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job = write_training_job(root, task="preprocess")
            remote_worker = worker.RemoteWorker(worker_args(job))
            remote_worker.tensor_dir.mkdir(parents=True, exist_ok=True)
            (remote_worker.tensor_dir / "old.pt").write_bytes(b"OLD dataset tensor fixture")
            stages: list[str] = []

            with (
                patch.object(remote_worker, "ensure_environment", return_value=remote_worker.env_paths()),
                patch.object(
                    remote_worker,
                    "tensor_cache_health",
                    return_value=(True, "tensor cache is finite", 1),
                ),
                patch.object(
                    remote_worker,
                    "run_command",
                    side_effect=lambda stage, command, cwd: stages.append(stage),
                ),
                patch("sys.stdout", new=StringIO()),
            ):
                status = remote_worker.train()

            result = json.loads(remote_worker.result_file.read_text(encoding="utf-8"))
            kinds = [item["kind"] for item in result["artifacts"]]
            self.assertEqual(status, 0)
            self.assertEqual(result["status"], "succeeded")
            self.assertNotIn("train", stages)
            self.assertEqual(stages, ["preprocess"])
            self.assertNotIn("final_adapter", kinds)
            self.assertIn("tensors", kinds)

    def test_tensor_reuse_requires_matching_dataset_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job = write_training_job(root, task="train_lora")
            remote_worker = worker.RemoteWorker(worker_args(job))
            remote_worker.tensor_dir.mkdir(parents=True, exist_ok=True)
            (remote_worker.tensor_dir / "old.pt").write_bytes(b"OLD dataset tensor fixture")
            (remote_worker.tensor_dir / "cache_fingerprint.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "variant": "stale",
                        "samples": [{"caption": "OLD caption", "audio_sha256": "deadbeef"}],
                    }
                ),
                encoding="utf-8",
            )
            stages: list[str] = []

            def run_command(stage: str, command: list[str], cwd: Path) -> None:
                stages.append(stage)
                if stage == "train":
                    final = remote_worker.output_dir / "final"
                    final.mkdir(parents=True, exist_ok=True)
                    (final / "adapter_model.safetensors").write_bytes(b"weights")

            with (
                patch.object(remote_worker, "ensure_environment", return_value=remote_worker.env_paths()),
                patch.object(
                    remote_worker,
                    "tensor_cache_health",
                    return_value=(True, "tensor cache is finite", 1),
                ),
                patch.object(remote_worker, "run_command", side_effect=run_command),
                patch("sys.stdout", new=StringIO()),
            ):
                status = remote_worker.train()

            self.assertEqual(status, 0)
            self.assertIn("preprocess", stages)
            self.assertIn("train", stages)
            self.assertFalse((remote_worker.tensor_dir / "old.pt").exists())

    def test_zero_seed_and_cfg_ratio_are_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job = write_training_job(root, training={"seed": 0, "cfg_ratio": 0.0})
            remote_worker = worker.RemoteWorker(worker_args(job))
            with patch("sys.stdout", new=StringIO()):
                remote_worker.make_sidestep_dataset()
                _, command = remote_worker.build_commands()

            self.assertEqual(command[command.index("--seed") + 1], "0")
            self.assertEqual(command[command.index("--cfg-ratio") + 1], "0.0")

    def test_training_without_adapter_weights_is_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job = write_training_job(root, task="train_lora")
            remote_worker = worker.RemoteWorker(worker_args(job))
            remote_worker.final_adapter_dir.mkdir(parents=True, exist_ok=True)
            leftover = remote_worker.final_adapter_dir / "old.safetensors"
            leftover.write_bytes(b"previous-run")
            stages: list[str] = []

            with (
                patch.object(remote_worker, "ensure_environment", return_value=remote_worker.env_paths()),
                patch.object(
                    remote_worker,
                    "tensor_cache_health",
                    return_value=(True, "tensor cache is finite", 1),
                ),
                patch.object(
                    remote_worker,
                    "run_command",
                    side_effect=lambda stage, command, cwd: stages.append(stage),
                ),
                patch("sys.stdout", new=StringIO()),
            ):
                status = remote_worker.train()

            result = json.loads(remote_worker.result_file.read_text(encoding="utf-8"))
            self.assertEqual(status, 1)
            self.assertEqual(result["status"], "failed")
            self.assertIn("train", stages)
            self.assertEqual(leftover.read_bytes(), b"previous-run")
