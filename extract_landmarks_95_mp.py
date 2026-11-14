"""Multiprocessing landmark extraction for [T, 95, 3] MediaPipe features.

This script is intentionally independent from PyTorch. It should be run from
the MediaPipe environment only.

Each worker process owns one MediaPipe Holistic instance and processes videos
independently. The default worker count is 8.
"""

from __future__ import annotations

import argparse
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import freeze_support
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

try:
    from .extract_landmarks_95 import (
        assert_video_feature_shape,
        extract_video_features,
        infer_split_name,
        iter_video_files,
        load_label_to_id,
        manifest_row,
        merge_all_manifest,
        parse_label_from_filename,
        write_manifest,
    )
except ImportError:
    from extract_landmarks_95 import (
        assert_video_feature_shape,
        extract_video_features,
        infer_split_name,
        iter_video_files,
        load_label_to_id,
        manifest_row,
        merge_all_manifest,
        parse_label_from_filename,
        write_manifest,
    )


_HOLISTIC = None


def init_worker(
    model_complexity: int,
    min_detection_confidence: float,
    min_tracking_confidence: float,
) -> None:
    """Create one MediaPipe Holistic instance per worker process."""
    import atexit
    import mediapipe as mp

    global _HOLISTIC
    _HOLISTIC = mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=model_complexity,
        smooth_landmarks=True,
        enable_segmentation=False,
        smooth_segmentation=True,
        refine_face_landmarks=False,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    atexit.register(_HOLISTIC.close)


def extract_one_video(task: dict[str, Any]) -> dict[str, Any]:
    """Worker-side extraction for one video."""
    if _HOLISTIC is None:
        raise RuntimeError("MediaPipe Holistic worker was not initialized")

    video_path = Path(task["video_path"])
    output_path = Path(task["output_path"])
    overwrite = bool(task["overwrite"])
    max_frames = task["max_frames"]

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        should_extract = overwrite or not output_path.exists()
        if not should_extract:
            existing = np.load(output_path, mmap_mode="r")
            assert_video_feature_shape(existing, output_path)
            status = "skipped"
        else:
            features = extract_video_features(video_path, _HOLISTIC, max_frames=max_frames)
            np.save(output_path, features)
            saved = np.load(output_path, mmap_mode="r")
            assert_video_feature_shape(saved, output_path)
            status = "written"

        return {
            "ok": True,
            "status": status,
            "video_path": str(video_path),
            "output_path": str(output_path),
            "row": task["row"],
        }
    except Exception as exc:
        return {
            "ok": False,
            "status": "failed",
            "video_path": str(video_path),
            "output_path": str(output_path),
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "row": task["row"],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract [T, 95, 3] MediaPipe features with multiprocessing.")
    parser.add_argument("--data_dir", type=Path, default=Path("data"), help="Directory containing videos.")
    parser.add_argument("--features_dir", type=Path, default=Path("features"), help="Output directory for .npy files.")
    parser.add_argument("--splits_dir", type=Path, default=Path("splits"), help="Output directory for CSV manifests.")
    parser.add_argument("--config_path", type=Path, default=Path("configs") / "labels.json", help="Path to labels.json.")
    parser.add_argument("--overwrite", action="store_true", help="Re-extract videos even if valid .npy files exist.")
    parser.add_argument("--skip_failed", action="store_true", help="Skip videos that fail extraction instead of stopping.")
    parser.add_argument(
        "--include_splits",
        nargs="*",
        default=None,
        help="Optional split/rep names to extract, e.g. rep1 rep4 rep3 generated2.",
    )
    parser.add_argument("--num_workers", type=int, default=8, help="Number of worker processes. Default: 8.")
    parser.add_argument("--max_frames", type=int, default=None, help="Optional maximum frames per video, mainly for debugging.")
    parser.add_argument("--model_complexity", type=int, default=2, choices=[0, 1, 2], help="MediaPipe Holistic model complexity.")   # was 2
    parser.add_argument("--min_detection_confidence", type=float, default=0.5)
    parser.add_argument("--min_tracking_confidence", type=float, default=0.5)
    return parser.parse_args()


def build_tasks(args: argparse.Namespace) -> tuple[list[dict[str, Any]], set[str] | None]:
    data_dir = args.data_dir.resolve()
    features_dir = args.features_dir.resolve()
    config_path = args.config_path.resolve()

    label_to_id = load_label_to_id(config_path)
    video_paths = iter_video_files(data_dir)
    if not video_paths:
        raise FileNotFoundError(f"No video files found under {data_dir}")

    include_splits = set(args.include_splits) if args.include_splits else None
    if include_splits is not None:
        video_paths = [path for path in video_paths if infer_split_name(path, data_dir) in include_splits]
        if not video_paths:
            raise FileNotFoundError(f"No videos found for include_splits={sorted(include_splits)} under {data_dir}")

    tasks: list[dict[str, Any]] = []
    for video_path in video_paths:
        row = manifest_row(video_path, data_dir, features_dir, label_to_id)
        output_path = features_dir / str(row["feature_path"])
        tasks.append(
            {
                "video_path": str(video_path.resolve()),
                "output_path": str(output_path.resolve()),
                "row": row,
                "overwrite": args.overwrite,
                "max_frames": args.max_frames,
            }
        )

    return tasks, include_splits


def write_outputs(
    splits_dir: Path,
    rows_by_split: dict[str, list[dict[str, str | int]]],
    all_rows: list[dict[str, str | int]],
    include_splits: set[str] | None,
    failures: list[dict[str, Any]],
) -> None:
    splits_dir.mkdir(parents=True, exist_ok=True)

    for split, rows in sorted(rows_by_split.items()):
        write_manifest(splits_dir / f"{split}.csv", rows)

    all_manifest_path = splits_dir / "all.csv"
    if include_splits is None:
        write_manifest(all_manifest_path, all_rows)
    else:
        merged_rows = merge_all_manifest(all_manifest_path, all_rows, replaced_splits=set(rows_by_split))
        write_manifest(all_manifest_path, merged_rows)

    if failures:
        failure_path = splits_dir / "failed_extractions.csv"
        write_manifest(
            failure_path,
            [
                {
                    "video_path": str(result["video_path"]),
                    "feature_path": str(result.get("output_path", "")),
                    "label": parse_label_from_filename(Path(result["video_path"]).name),
                    "label_id": "",
                    "split": str(result["error"]),
                }
                for result in failures
            ],
        )
        print(f"Skipped {len(failures)} failed videos. See {failure_path}")


def main() -> None:
    args = parse_args()
    if args.num_workers < 1:
        raise ValueError("--num_workers must be >= 1")

    splits_dir = args.splits_dir.resolve()
    tasks, include_splits = build_tasks(args)

    print(f"Found {len(tasks)} videos to process")
    print(f"Using {args.num_workers} worker processes")

    rows_by_split: dict[str, list[dict[str, str | int]]] = {}
    all_rows: list[dict[str, str | int]] = []
    failures: list[dict[str, Any]] = []
    written = 0
    skipped = 0

    with ProcessPoolExecutor(
        max_workers=args.num_workers,
        initializer=init_worker,
        initargs=(
            args.model_complexity,
            args.min_detection_confidence,
            args.min_tracking_confidence,
        ),
    ) as executor:
        futures = [executor.submit(extract_one_video, task) for task in tasks]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting landmarks", unit="video"):
            result = future.result()
            if not result["ok"]:
                if not args.skip_failed:
                    print(result["traceback"])
                    raise RuntimeError(f"Failed extracting {result['video_path']}: {result['error']}")
                failures.append(result)
                continue

            row = result["row"]
            split = str(row["split"])
            rows_by_split.setdefault(split, []).append(row)
            all_rows.append(row)

            if result["status"] == "written":
                written += 1
            else:
                skipped += 1

    write_outputs(splits_dir, rows_by_split, all_rows, include_splits, failures)

    print(f"Wrote {written} feature files")
    print(f"Skipped {skipped} existing valid feature files")
    print(f"Wrote {len(all_rows)} manifest rows under {splits_dir}")


if __name__ == "__main__":
    freeze_support()
    main()
