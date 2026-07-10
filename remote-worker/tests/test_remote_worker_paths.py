from __future__ import annotations

import argparse
import importlib.util
from importlib.machinery import SourceFileLoader
import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch


WORKER_PATH = Path(__file__).resolve().parents[1] / "pimp-remote-worker"
LOADER = SourceFileLoader("pimp_remote_worker", str(WORKER_PATH))
SPEC = importlib.util.spec_from_loader(LOADER.name, LOADER)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not load remote worker from {WORKER_PATH}")
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)


def write_job(root: Path, *, package_root: str = ".", outputs: dict[str, str] | None = None) -> Path:
    job = {
        "job_id": "path-test",
        "task": "train_lokr",
        "package": {
            "package_root": package_root,
            "entry_bundle": "training_bundle.json",
        },
        "outputs": outputs or {
            "logs_file": "remote_artifacts/logs.jsonl",
            "metrics_file": "remote_artifacts/metrics.jsonl",
            "result_file": "remote_artifacts/result.json",
            "checkpoints_directory": "remote_artifacts/checkpoints",
            "final_adapter_directory": "remote_artifacts/final_adapter",
        },
    }
    path = root / "remote_job.json"
    path.write_text(json.dumps(job), encoding="utf-8")
    return path


def worker_args(job: Path) -> argparse.Namespace:
    return argparse.Namespace(job=str(job), dry_run=True, no_download=True)


class RemoteWorkerPathTests(unittest.TestCase):
    def test_package_root_must_remain_at_remote_job_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job = write_job(root, package_root="../outside")

            with self.assertRaisesRegex(ValueError, "package.package_root"):
                worker.RemoteWorker(worker_args(job))

    def test_output_paths_cannot_escape_or_target_package_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            escaping = write_job(
                root,
                outputs={
                    "logs_file": "remote_artifacts/logs.jsonl",
                    "metrics_file": "remote_artifacts/metrics.jsonl",
                    "result_file": "remote_artifacts/result.json",
                    "checkpoints_directory": "remote_artifacts/checkpoints",
                    "final_adapter_directory": "../../outside",
                },
            )
            with self.assertRaisesRegex(ValueError, "final_adapter_directory"):
                worker.RemoteWorker(worker_args(escaping))

            targeting_root = write_job(
                root,
                outputs={
                    "logs_file": "remote_artifacts/logs.jsonl",
                    "metrics_file": "remote_artifacts/metrics.jsonl",
                    "result_file": "remote_artifacts/result.json",
                    "checkpoints_directory": ".",
                    "final_adapter_directory": "remote_artifacts/final_adapter",
                },
            )
            with self.assertRaisesRegex(ValueError, "checkpoints_directory"):
                worker.RemoteWorker(worker_args(targeting_root))

    def test_destructive_outputs_cannot_overlap_artifacts_or_staging(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job = write_job(
                root,
                outputs={
                    "logs_file": "remote_artifacts/logs.jsonl",
                    "metrics_file": "remote_artifacts/metrics.jsonl",
                    "result_file": "remote_artifacts/result.json",
                    "checkpoints_directory": "remote_artifacts/checkpoints",
                    "final_adapter_directory": "remote_artifacts/side_step_output",
                },
            )

            with self.assertRaisesRegex(ValueError, "overlaps worker staging"):
                worker.RemoteWorker(worker_args(job))

    def test_output_files_cannot_live_inside_replaceable_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job = write_job(
                root,
                outputs={
                    "logs_file": "remote_artifacts/final_adapter/logs.jsonl",
                    "metrics_file": "remote_artifacts/metrics.jsonl",
                    "result_file": "remote_artifacts/result.json",
                    "checkpoints_directory": "remote_artifacts/checkpoints",
                    "final_adapter_directory": "remote_artifacts/final_adapter",
                },
            )

            with self.assertRaisesRegex(ValueError, "overlaps replaceable directory"):
                worker.RemoteWorker(worker_args(job))

    def test_entry_bundle_must_stay_inside_package(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job = write_job(root)
            payload = json.loads(job.read_text(encoding="utf-8"))
            payload["package"]["entry_bundle"] = "../outside.json"
            job.write_text(json.dumps(payload), encoding="utf-8")
            remote_worker = worker.RemoteWorker(worker_args(job))

            with self.assertRaisesRegex(ValueError, "package.entry_bundle"):
                remote_worker.selected_bundle_path()

    def test_clip_audio_must_stay_inside_package(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            container = Path(temporary)
            root = container / "package"
            root.mkdir()
            outside_audio = container / "outside.wav"
            outside_audio.write_bytes(b"outside")
            job = write_job(root)
            (root / "training_bundle.json").write_text(
                json.dumps({"clips": [{"id": "escape", "file": "../outside.wav"}]}),
                encoding="utf-8",
            )
            remote_worker = worker.RemoteWorker(worker_args(job))

            with self.assertRaisesRegex(ValueError, r"clip\[0\].audio_path"):
                remote_worker.make_sidestep_dataset()

            self.assertEqual(outside_audio.read_bytes(), b"outside")

    def test_zip_extraction_rejects_parent_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive_path = root / "malicious.zip"
            extraction_root = root / "jobs"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("remote_job.json", "{}")
                archive.writestr("../escaped.txt", "must not escape")

            with patch.dict(os.environ, {"PIMP_REMOTE_JOB_DIR": str(extraction_root)}):
                with self.assertRaisesRegex(ValueError, "Zip member escapes"):
                    worker.find_remote_job(archive_path)

            self.assertFalse((root / "escaped.txt").exists())


if __name__ == "__main__":
    unittest.main()
