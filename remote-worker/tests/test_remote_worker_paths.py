from __future__ import annotations

import argparse
import importlib.util
from importlib.machinery import SourceFileLoader
import json
import os
import stat
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


def write_package_v2_job(root: Path, *, include: list[str] | None = None) -> Path:
    selected_roles = include or ["full_mix"]
    job = write_job(root)
    payload = json.loads(job.read_text(encoding="utf-8"))
    payload["package"].update(
        {
            "dataset_manifest": "dataset_manifest.json",
            "selected_bundle": {
                "name": "full_mix",
                "path": "training_bundle.json",
                "include": selected_roles,
            },
        }
    )
    job.write_text(json.dumps(payload), encoding="utf-8")
    (root / "sources").mkdir()
    (root / "stems").mkdir()
    (root / "sources" / "song.wav").write_bytes(b"full mix")
    (root / "stems" / "vocals.wav").write_bytes(b"vocals")
    manifest = {
        "pimp_audio_dataset_version": "2.0",
        "source_tracks": [
            {
                "source_track_id": "track-1",
                "original_file_name": "song.wav",
                "duration_seconds": 300,
            }
        ],
        "assets": [
            {
                "asset_id": "asset-full",
                "source_track_id": "track-1",
                "asset_role": "full_mix",
                "file": {"path": "sources/song.wav", "duration_seconds": 300},
                "source_range": {"start_seconds": 0, "end_seconds": 300},
                "annotation": {
                    "caption": {"text": "reviewed whole-track caption", "source": "native"},
                    "lyrics": {"text": "[Verse]\\nReviewed lyrics", "source": "native", "language": "en"},
                    "review_status": "approved",
                    "approved": True,
                },
            },
            {
                "asset_id": "asset-vocals",
                "source_track_id": "track-1",
                "asset_role": "vocals",
                "file": {"path": "stems/vocals.wav", "duration_seconds": 300},
                "source_range": {"start_seconds": 0, "end_seconds": 300},
                "annotation": {
                    "caption": {"text": "reviewed vocal stem", "source": "native"},
                    "lyrics": {"text": "[Verse]\\nReviewed lyrics", "source": "native"},
                    "review_status": "approved",
                    "approved": True,
                },
            },
        ],
        "training_plans": [
            {
                "id": "ace-step-whole-track",
                "backend_id": "ace-step-1.5",
                "name": "Whole tracks",
                "is_default": True,
                "selections": [
                    {
                        "asset_id": "asset-full",
                        "include_in_training": True,
                        "training_range": {"start_seconds": 0, "end_seconds": 240},
                        "range_policy": "backend_limit",
                    },
                    {
                        "asset_id": "asset-vocals",
                        "include_in_training": False,
                        "training_range": {"start_seconds": 20, "end_seconds": 140},
                        "range_policy": "manual",
                    },
                ],
            }
        ],
    }
    (root / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    bundle = {
        "source_package_version": "2.0",
        "training_plan_id": "ace-step-whole-track",
        "clips": [
            {
                "id": "asset-full",
                "asset_id": "asset-full",
                "asset_role": "full_mix",
                "file": "sources/song.wav",
                "include_in_training": True,
            },
            {
                "id": "asset-vocals",
                "asset_id": "asset-vocals",
                "asset_role": "vocals",
                "file": "stems/vocals.wav",
                "include_in_training": True,
            },
        ],
    }
    (root / "training_bundle.json").write_text(json.dumps(bundle), encoding="utf-8")
    return job


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

    def test_zip_extraction_rejects_symbolic_links(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive_path = root / "symlink.zip"
            extraction_root = root / "jobs"
            link = zipfile.ZipInfo("package/escaped-link")
            link.create_system = 3
            link.external_attr = (stat.S_IFLNK | 0o777) << 16
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("package/remote_job.json", "{}")
                archive.writestr(link, "../../escaped.txt")

            with patch.dict(os.environ, {"PIMP_REMOTE_JOB_DIR": str(extraction_root)}):
                with self.assertRaisesRegex(ValueError, "symbolic link"):
                    worker.find_remote_job(archive_path)

            self.assertFalse(extraction_root.exists())

    def test_package_v2_selects_assets_by_role_and_preserves_annotations_and_ranges(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job = write_package_v2_job(root)
            remote_worker = worker.RemoteWorker(worker_args(job))

            dataset = remote_worker.make_sidestep_dataset()

            self.assertEqual(dataset["metadata"]["source_package_version"], "2.0")
            self.assertEqual(dataset["metadata"]["training_plan_id"], "ace-step-whole-track")
            self.assertEqual(dataset["metadata"]["selected_asset_roles"], ["full_mix"])
            self.assertEqual(len(dataset["samples"]), 1)
            sample = dataset["samples"][0]
            self.assertEqual(sample["asset_id"], "asset-full")
            self.assertEqual(sample["asset_role"], "full_mix")
            self.assertEqual(sample["caption"], "reviewed whole-track caption")
            self.assertEqual(sample["lyrics"], "[Verse]\\nReviewed lyrics")
            self.assertEqual(sample["training_range"], {"start_seconds": 0, "end_seconds": 240})
            self.assertEqual(sample["range_policy"], "backend_limit")
            self.assertEqual(sample["duration"], 240)
            self.assertEqual(
                remote_worker.dataset_selection["training_ranges"],
                [
                    {
                        "asset_id": "asset-full",
                        "start_seconds": 0,
                        "end_seconds": 240,
                        "range_policy": "backend_limit",
                    }
                ],
            )

    def test_package_v2_explicit_stem_role_uses_plan_range_without_defaulting_to_mix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job = write_package_v2_job(root, include=["vocals"])
            remote_worker = worker.RemoteWorker(worker_args(job))

            dataset = remote_worker.make_sidestep_dataset()

            self.assertEqual(len(dataset["samples"]), 1)
            sample = dataset["samples"][0]
            self.assertEqual(sample["asset_id"], "asset-vocals")
            self.assertEqual(sample["asset_role"], "vocals")
            self.assertEqual(sample["training_range"], {"start_seconds": 20, "end_seconds": 140})
            self.assertEqual(sample["duration"], 120)

    def test_package_v2_rejects_training_ranges_above_ace_step_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job = write_package_v2_job(root)
            manifest_path = root / "dataset_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["training_plans"][0]["selections"][0]["training_range"] = {
                "start_seconds": 0,
                "end_seconds": 250,
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            remote_worker = worker.RemoteWorker(worker_args(job))

            with self.assertRaisesRegex(ValueError, "240 second limit"):
                remote_worker.make_sidestep_dataset()

    def test_package_v2_dry_run_writes_selection_and_stable_result_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job = write_package_v2_job(root)
            remote_worker = worker.RemoteWorker(worker_args(job))

            exit_code = remote_worker.train()

            self.assertEqual(exit_code, 0)
            dry_run = json.loads(
                (root / "remote_artifacts" / "command" / "dry_run.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                dry_run["package_v2_selection"]["asset_ids"],
                ["asset-full"],
            )
            self.assertEqual(dry_run["sample_count"], 1)
            result = json.loads(
                (root / "remote_artifacts" / "result.json").read_text(encoding="utf-8")
            )
            self.assertEqual(result["pimp_audio_remote_result_version"], "1.0")
            self.assertEqual(result["status"], "succeeded")
            self.assertEqual(result["metrics"]["dry_run"], 1.0)

    def test_exported_package_v2_zip_completes_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            container = Path(temporary)
            package_root = container / "reviewed-package"
            package_root.mkdir()
            write_package_v2_job(package_root)
            archive_path = container / "reviewed-package.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                for source in package_root.rglob("*"):
                    if source.is_file():
                        archive.write(
                            source,
                            Path(package_root.name) / source.relative_to(package_root),
                        )

            extraction_root = container / "remote-jobs"
            with patch.dict(os.environ, {"PIMP_REMOTE_JOB_DIR": str(extraction_root)}):
                remote_worker = worker.RemoteWorker(worker_args(archive_path))
                exit_code = remote_worker.train()

            self.assertEqual(exit_code, 0)
            self.assertTrue(
                remote_worker.package_root.is_relative_to(extraction_root.resolve())
            )
            result = json.loads(
                (remote_worker.package_root / "remote_artifacts" / "result.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(result["pimp_audio_remote_result_version"], "1.0")
            self.assertEqual(result["status"], "succeeded")
            self.assertEqual(result["metrics"]["sample_count"], 1.0)


if __name__ == "__main__":
    unittest.main()
