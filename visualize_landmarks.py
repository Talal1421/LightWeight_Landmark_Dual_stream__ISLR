"""Draw raw visual landmarks from extracted [T, 95, 3] features onto videos.

The 95 features are:
  - 63 visual landmarks: left hand 21 + right hand 21 + upper-body pose 21
  - 32 derived/non-visual features: positions, angles, curls, velocities

This script draws only the 63 raw visual landmarks.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
EXPECTED_FEATURE_SHAPE = (95, 3)

LEFT_HAND_SLICE = slice(0, 21)
RIGHT_HAND_SLICE = slice(35, 56)
POSE_SLICE = slice(70, 91)

HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
]

POSE_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (0, 4),
    (4, 5),
    (5, 6),
    (3, 7),
    (6, 8),
    (9, 10),
    (11, 12),
    (11, 13),
    (13, 15),
    (15, 17),
    (15, 19),
    (17, 19),
    (12, 14),
    (14, 16),
    (16, 18),
    (16, 20),
    (18, 20),
]


def iter_video_files(video_dir: Path) -> list[Path]:
    return sorted(
        (path for path in video_dir.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS),
        key=lambda path: str(path).lower(),
    )


def assert_feature_array(features: np.ndarray, feature_path: Path) -> None:
    if features.ndim != 3 or features.shape[1:] != EXPECTED_FEATURE_SHAPE:
        raise AssertionError(f"Expected {feature_path} to have shape [T, 95, 3], got {features.shape}")
    if features.dtype != np.float32:
        raise AssertionError(f"Expected {feature_path} to use float32, got {features.dtype}")


def feature_candidates(video_path: Path, video_dir: Path, features_dir: Path) -> list[Path]:
    """Return likely .npy paths for a video."""
    relative_to_video_dir = video_path.relative_to(video_dir).with_suffix(".npy")
    candidates = [
        features_dir / video_dir.name / relative_to_video_dir,
        features_dir / relative_to_video_dir,
    ]

    matches = sorted(features_dir.rglob(f"{video_path.stem}.npy"), key=lambda path: str(path).lower())
    candidates.extend(matches)

    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen:
            unique.append(candidate)
            seen.add(resolved)
    return unique


def find_feature_path(video_path: Path, video_dir: Path, features_dir: Path) -> Path | None:
    for candidate in feature_candidates(video_path, video_dir, features_dir):
        if candidate.exists():
            return candidate.resolve()
    return None


def is_visible_point(point: np.ndarray) -> bool:
    return bool(np.any(np.abs(point) > 1e-8))


def to_pixel(point: np.ndarray, width: int, height: int) -> tuple[int, int]:
    x = int(np.clip(point[0], 0.0, 1.0) * (width - 1))
    y = int(np.clip(point[1], 0.0, 1.0) * (height - 1))
    return x, y


def draw_landmark_group(
    frame: np.ndarray,
    landmarks: np.ndarray,
    connections: list[tuple[int, int]],
    point_color: tuple[int, int, int],
    line_color: tuple[int, int, int],
    radius: int,
    thickness: int,
) -> None:
    height, width = frame.shape[:2]
    visible = [is_visible_point(point) for point in landmarks]
    pixels = [to_pixel(point, width, height) if is_visible_point(point) else None for point in landmarks]

    for start, end in connections:
        if visible[start] and visible[end] and pixels[start] is not None and pixels[end] is not None:
            cv2.line(frame, pixels[start], pixels[end], line_color, thickness, lineType=cv2.LINE_AA)

    for point in pixels:
        if point is not None:
            cv2.circle(frame, point, radius, point_color, -1, lineType=cv2.LINE_AA)


def output_path_for(video_path: Path, video_dir: Path, output_dir: Path) -> Path:
    relative = video_path.relative_to(video_dir)
    return (output_dir / video_dir.name / relative).with_name(f"{relative.stem}_landmarks.mp4")


def visualize_video(
    video_path: Path,
    feature_path: Path,
    output_path: Path,
    overwrite: bool,
    radius: int,
    thickness: int,
) -> bool:
    if output_path.exists() and not overwrite:
        return False

    features = np.load(feature_path)
    assert_feature_array(features, feature_path)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 25.0

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create output video: {output_path}")

    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            if frame_index < features.shape[0]:
                frame_features = features[frame_index]
                draw_landmark_group(
                    frame,
                    frame_features[LEFT_HAND_SLICE],
                    HAND_CONNECTIONS,
                    point_color=(0, 255, 0),
                    line_color=(0, 150, 0),
                    radius=radius,
                    thickness=thickness,
                )
                draw_landmark_group(
                    frame,
                    frame_features[RIGHT_HAND_SLICE],
                    HAND_CONNECTIONS,
                    point_color=(255, 180, 0),
                    line_color=(180, 100, 0),
                    radius=radius,
                    thickness=thickness,
                )
                draw_landmark_group(
                    frame,
                    frame_features[POSE_SLICE],
                    POSE_CONNECTIONS,
                    point_color=(0, 140, 255),
                    line_color=(0, 80, 220),
                    radius=radius,
                    thickness=thickness,
                )

            writer.write(frame)
            frame_index += 1
    finally:
        capture.release()
        writer.release()

    if frame_index != features.shape[0]:
        print(f"Warning: {video_path.name} has {frame_index} video frames but {features.shape[0]} feature frames")

    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize the 63 raw landmarks on videos.")
    parser.add_argument("--video_dir", type=Path, required=True, help="Directory containing videos to visualize.")
    parser.add_argument("--features_dir", type=Path, default=Path("features"), help="Root directory containing .npy features.")
    parser.add_argument("--output_dir", type=Path, default=Path("vis_landmarks"), help="Directory for output videos.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing visualized videos.")
    parser.add_argument("--max_videos", type=int, default=None, help="Optional limit for quick checks.")
    parser.add_argument("--radius", type=int, default=3, help="Landmark dot radius.")
    parser.add_argument("--thickness", type=int, default=2, help="Connection line thickness.")
    parser.add_argument("--skip_missing", action="store_true", help="Skip videos whose .npy feature file is missing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_dir = args.video_dir.resolve()
    features_dir = args.features_dir.resolve()
    output_dir = args.output_dir.resolve()

    video_paths = iter_video_files(video_dir)
    if args.max_videos is not None:
        video_paths = video_paths[: args.max_videos]
    if not video_paths:
        raise FileNotFoundError(f"No videos found under {video_dir}")

    print("Drawing 63 visual landmarks: left hand 21 + right hand 21 + pose 21")
    print("Skipping 32 derived/non-visual features: hand positions/angles/curls + pose positions/velocities")

    written = 0
    skipped = 0
    for video_path in tqdm(video_paths, desc="visualizing", unit="video"):
        feature_path = find_feature_path(video_path, video_dir, features_dir)
        if feature_path is None:
            message = f"No feature file found for {video_path}"
            if args.skip_missing:
                print(f"Warning: {message}")
                skipped += 1
                continue
            raise FileNotFoundError(message)

        output_path = output_path_for(video_path, video_dir, output_dir)
        did_write = visualize_video(
            video_path=video_path,
            feature_path=feature_path,
            output_path=output_path,
            overwrite=args.overwrite,
            radius=args.radius,
            thickness=args.thickness,
        )
        if did_write:
            written += 1
        else:
            skipped += 1

    print(f"Wrote {written} visualized videos to {output_dir}")
    if skipped:
        print(f"Skipped {skipped} videos")


if __name__ == "__main__":
    main()
