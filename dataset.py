"""Dataset utilities for pre-extracted [T, 95, 3] feature arrays."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


EXPECTED_FEATURE_SHAPE = (95, 3)
NORMALIZATION_CHOICES = {"none", "body"}
FEATURE_SET_CHOICES = {"full", "no_face"}

LEFT_HAND_RAW = slice(0, 21)
LEFT_HAND_POSITIONS = slice(21, 26)
LEFT_HAND_ANGLES_AND_CURLS = slice(26, 35)

RIGHT_HAND_RAW = slice(35, 56)
RIGHT_HAND_POSITIONS = slice(56, 61)
RIGHT_HAND_ANGLES_AND_CURLS = slice(61, 70)

POSE_RAW = slice(70, 91)
POSE_POSITIONS_AND_VELOCITIES = slice(91, 95)

POSE_LEFT_SHOULDER = 11
POSE_RIGHT_SHOULDER = 12

NO_FACE_FEATURE_INDICES = np.array(
    [
        *range(0, 70),  # both hands, including derived hand features
        *range(81, 91),  # pose raw landmarks 11-20: shoulders, elbows, wrists, pinky/index
        *range(93, 95),  # wrist velocity features; drops hand-to-face position features
    ],
    dtype=np.int64,
)
FEATURE_SET_SIZES = {
    "full": 95,
    "no_face": int(NO_FACE_FEATURE_INDICES.size),
}


def assert_feature_array(array: np.ndarray, path: Path) -> None:
    if array.ndim != 3 or array.shape[1:] != EXPECTED_FEATURE_SHAPE:
        raise AssertionError(f"Expected {path} to have shape [T, 95, 3], got {array.shape}")
    if array.dtype != np.float32:
        raise AssertionError(f"Expected {path} to use float32, got {array.dtype}")
    if array.shape[0] <= 0:
        raise AssertionError(f"Expected {path} to contain at least one frame")


def _nonzero_rows(rows: np.ndarray) -> np.ndarray:
    return np.any(np.abs(rows) > 1e-8, axis=1)


def _normalize_raw_landmarks(
    raw_landmarks: np.ndarray,
    centers: np.ndarray,
    scales: np.ndarray,
    center_z: bool,
) -> np.ndarray:
    normalized = raw_landmarks.copy()
    present = np.any(np.abs(raw_landmarks) > 1e-8, axis=2)
    normalized[:, :, :2] = (normalized[:, :, :2] - centers[:, None, :2]) / scales[:, None, None]
    if center_z:
        normalized[:, :, 2] = (normalized[:, :, 2] - centers[:, None, 2]) / scales[:, None]
    else:
        normalized[:, :, 2] = normalized[:, :, 2] / scales[:, None]
    normalized[~present] = 0.0
    return normalized


def _scale_relative_features(relative_features: np.ndarray, scales: np.ndarray) -> np.ndarray:
    normalized = relative_features.copy()
    present = np.any(np.abs(relative_features) > 1e-8, axis=2)
    normalized = normalized / scales[:, None, None]
    normalized[~present] = 0.0
    return normalized


def normalize_body(array: np.ndarray) -> np.ndarray:
    """Normalize landmarks around the shoulder midpoint and shoulder width.

    Raw visual landmarks are centered by the midpoint between left/right
    shoulders and divided by shoulder distance. Relative position/velocity
    features are divided by the same scale. Angle and curl features are left
    unchanged because they are already scale-invariant.
    """
    normalized = array.copy()
    pose = array[:, POSE_RAW, :]
    left_shoulder = pose[:, POSE_LEFT_SHOULDER, :]
    right_shoulder = pose[:, POSE_RIGHT_SHOULDER, :]

    shoulders_present = _nonzero_rows(left_shoulder) & _nonzero_rows(right_shoulder)
    centers = (left_shoulder + right_shoulder) / 2.0
    scales = np.linalg.norm(left_shoulder[:, :2] - right_shoulder[:, :2], axis=1)

    valid = shoulders_present & (scales > 1e-6)
    centers = centers.astype(np.float32)
    scales = scales.astype(np.float32)
    centers[~valid] = 0.0
    scales[~valid] = 1.0

    normalized[:, LEFT_HAND_RAW, :] = _normalize_raw_landmarks(array[:, LEFT_HAND_RAW, :], centers, scales, center_z=False)
    normalized[:, RIGHT_HAND_RAW, :] = _normalize_raw_landmarks(array[:, RIGHT_HAND_RAW, :], centers, scales, center_z=False)
    normalized[:, POSE_RAW, :] = _normalize_raw_landmarks(array[:, POSE_RAW, :], centers, scales, center_z=True)

    normalized[:, LEFT_HAND_POSITIONS, :] = _scale_relative_features(array[:, LEFT_HAND_POSITIONS, :], scales)
    normalized[:, RIGHT_HAND_POSITIONS, :] = _scale_relative_features(array[:, RIGHT_HAND_POSITIONS, :], scales)
    normalized[:, POSE_POSITIONS_AND_VELOCITIES, :] = _scale_relative_features(
        array[:, POSE_POSITIONS_AND_VELOCITIES, :],
        scales,
    )

    normalized[:, LEFT_HAND_ANGLES_AND_CURLS, :] = array[:, LEFT_HAND_ANGLES_AND_CURLS, :]
    normalized[:, RIGHT_HAND_ANGLES_AND_CURLS, :] = array[:, RIGHT_HAND_ANGLES_AND_CURLS, :]
    return normalized.astype(np.float32)


def apply_normalization(array: np.ndarray, normalization: str) -> np.ndarray:
    if normalization not in NORMALIZATION_CHOICES:
        raise ValueError(f"Unknown normalization {normalization!r}. Expected one of {sorted(NORMALIZATION_CHOICES)}")
    if normalization == "none":
        return array
    if normalization == "body":
        return normalize_body(array)
    raise AssertionError(f"Unhandled normalization: {normalization}")


def apply_feature_set(array: np.ndarray, feature_set: str) -> np.ndarray:
    if feature_set not in FEATURE_SET_CHOICES:
        raise ValueError(f"Unknown feature_set {feature_set!r}. Expected one of {sorted(FEATURE_SET_CHOICES)}")
    if feature_set == "full":
        return array
    if feature_set == "no_face":
        return array[:, NO_FACE_FEATURE_INDICES, :]
    raise AssertionError(f"Unhandled feature_set: {feature_set}")


def build_missing_landmark_mask(array: np.ndarray) -> np.ndarray:
    """Return a block-aware present/missing channel for MediaPipe detections.

    A hand or pose block is considered missing only when its full raw landmark
    block is all zeros. Derived features for the same block receive the same
    mask value so angle slots such as [angle, 0, 0] are not treated as missing.
    """
    mask = np.ones((array.shape[0], 95, 1), dtype=np.float32)

    left_missing = ~np.any(np.abs(array[:, LEFT_HAND_RAW, :]) > 1e-8, axis=(1, 2))
    right_missing = ~np.any(np.abs(array[:, RIGHT_HAND_RAW, :]) > 1e-8, axis=(1, 2))
    pose_missing = ~np.any(np.abs(array[:, POSE_RAW, :]) > 1e-8, axis=(1, 2))

    mask[left_missing, 0:35, :] = 0.0
    mask[right_missing, 35:70, :] = 0.0
    mask[pose_missing, 70:95, :] = 0.0
    return mask


class SignLanguageFeatureDataset(Dataset):
    """Loads feature tensors and labels from a CSV manifest."""

    def __init__(
        self,
        csv_path: str | Path,
        features_dir: str | Path | None = None,
        normalization: str = "none",
        feature_set: str = "full",
        add_missing_mask: bool = False,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.features_dir = Path(features_dir) if features_dir is not None else None
        if normalization not in NORMALIZATION_CHOICES:
            raise ValueError(f"Unknown normalization {normalization!r}. Expected one of {sorted(NORMALIZATION_CHOICES)}")
        if feature_set not in FEATURE_SET_CHOICES:
            raise ValueError(f"Unknown feature_set {feature_set!r}. Expected one of {sorted(FEATURE_SET_CHOICES)}")
        self.normalization = normalization
        self.feature_set = feature_set
        self.add_missing_mask = bool(add_missing_mask)
        self.feature_shape = (FEATURE_SET_SIZES[feature_set], 4 if self.add_missing_mask else 3)
        self.frame = pd.read_csv(self.csv_path)

        required_columns = {"feature_path", "label_id"}
        missing = required_columns.difference(self.frame.columns)
        if missing:
            raise ValueError(f"{self.csv_path} is missing columns: {sorted(missing)}")
        if self.frame.empty:
            raise ValueError(f"{self.csv_path} contains no rows")

    def __len__(self) -> int:
        return len(self.frame)

    def _resolve_feature_path(self, feature_path: str) -> Path:
        path = Path(feature_path)
        if path.is_absolute():
            return path
        if self.features_dir is None:
            return (self.csv_path.parent / path).resolve()
        return (self.features_dir / path).resolve()

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        row = self.frame.iloc[index]
        feature_path = self._resolve_feature_path(str(row["feature_path"]))
        array = np.load(feature_path)
        assert_feature_array(array, feature_path)
        missing_mask = build_missing_landmark_mask(array) if self.add_missing_mask else None
        array = apply_normalization(array, self.normalization)
        array = apply_feature_set(array, self.feature_set)
        if missing_mask is not None:
            missing_mask = apply_feature_set(missing_mask, self.feature_set)
            array = np.concatenate([array, missing_mask], axis=2).astype(np.float32)

        features = torch.from_numpy(array).float()
        length = int(features.shape[0])
        label = int(row["label_id"])
        return features, length, label


def collate_fn(batch: list[tuple[torch.Tensor, int, int]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([item[1] for item in batch], dtype=torch.long)
    labels = torch.tensor([item[2] for item in batch], dtype=torch.long)
    max_length = int(lengths.max().item())
    feature_count, coordinate_count = batch[0][0].shape[1:]

    padded = torch.zeros((len(batch), max_length, feature_count, coordinate_count), dtype=torch.float32)
    for batch_index, (features, length, _) in enumerate(batch):
        padded[batch_index, :length] = features

    return padded, lengths, labels
