#!/usr/bin/env python3
"""Shared, dependency-light trajectory evaluation primitives.

Quaternions use the TUM ``x y z w`` convention.  Sparse-survey evaluation is
deliberately truth-sample driven: estimates may be interpolated to an observed
survey timestamp, but truth is never synthesized between survey observations.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


def load_trajectory(path: Path) -> np.ndarray:
    values = np.loadtxt(path, dtype=np.float64, comments="#")
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.shape[1] < 4:
        raise ValueError(f"{path} must contain timestamp x y z columns")
    if not np.all(np.isfinite(values[:, :4])):
        raise ValueError(f"{path} contains non-finite timestamp or position")
    order = np.argsort(values[:, 0], kind="stable")
    values = values[order]
    if np.any(np.diff(values[:, 0]) <= 0.0):
        raise ValueError(f"{path} timestamps must be unique and strictly increasing")
    return values


def normalize_quaternions(quaternions: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(quaternions, axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms < 1e-12):
        raise ValueError("trajectory contains an invalid quaternion")
    return quaternions / norms[:, None]


def quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Multiply arrays of xyzw quaternions with numpy broadcasting."""
    lx, ly, lz, lw = np.moveaxis(left, -1, 0)
    rx, ry, rz, rw = np.moveaxis(right, -1, 0)
    return np.stack((
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    ), axis=-1)


def quaternion_conjugate(quaternions: np.ndarray) -> np.ndarray:
    result = np.array(quaternions, dtype=np.float64, copy=True)
    result[..., :3] *= -1.0
    return result


def rotate_vectors(quaternions: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    q = normalize_quaternions(np.asarray(quaternions, dtype=np.float64))
    xyz = q[:, :3]
    w = q[:, 3:4]
    vectors = np.asarray(vectors, dtype=np.float64)
    return vectors + 2.0 * np.cross(xyz, np.cross(xyz, vectors) + w * vectors)


def inverse_rotate_vectors(quaternions: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    return rotate_vectors(quaternion_conjugate(normalize_quaternions(quaternions)),
                          vectors)


def quaternion_slerp(left: np.ndarray, right: np.ndarray,
                     alpha: np.ndarray) -> np.ndarray:
    left = normalize_quaternions(left)
    right = normalize_quaternions(right)
    dot = np.sum(left * right, axis=1)
    flip = dot < 0.0
    right = right.copy()
    right[flip] *= -1.0
    dot = np.clip(np.abs(dot), -1.0, 1.0)
    result = np.empty_like(left)
    linear = dot > 0.9995
    result[linear] = ((1.0 - alpha[linear, None]) * left[linear] +
                      alpha[linear, None] * right[linear])
    curved = ~linear
    if np.any(curved):
        theta = np.arccos(dot[curved])
        denominator = np.sin(theta)
        result[curved] = (
            np.sin((1.0 - alpha[curved]) * theta)[:, None] /
            denominator[:, None] * left[curved] +
            np.sin(alpha[curved] * theta)[:, None] /
            denominator[:, None] * right[curved])
    return normalize_quaternions(result)


def rotation_matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    """Convert one proper 3x3 rotation matrix to a normalized xyzw quaternion."""
    matrix = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.array([
            (matrix[2, 1] - matrix[1, 2]) / scale,
            (matrix[0, 2] - matrix[2, 0]) / scale,
            (matrix[1, 0] - matrix[0, 1]) / scale,
            0.25 * scale,
        ])
    else:
        axis = int(np.argmax(np.diag(matrix)))
        if axis == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quaternion = np.array([0.25 * scale,
                                   (matrix[0, 1] + matrix[1, 0]) / scale,
                                   (matrix[0, 2] + matrix[2, 0]) / scale,
                                   (matrix[2, 1] - matrix[1, 2]) / scale])
        elif axis == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quaternion = np.array([(matrix[0, 1] + matrix[1, 0]) / scale,
                                   0.25 * scale,
                                   (matrix[1, 2] + matrix[2, 1]) / scale,
                                   (matrix[0, 2] - matrix[2, 0]) / scale])
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quaternion = np.array([(matrix[0, 2] + matrix[2, 0]) / scale,
                                   (matrix[1, 2] + matrix[2, 1]) / scale,
                                   0.25 * scale,
                                   (matrix[1, 0] - matrix[0, 1]) / scale])
    return quaternion / np.linalg.norm(quaternion)


def align_se3(source: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("SE(3) alignment requires matching Nx3 point arrays")
    if source.shape[0] < 3:
        raise ValueError("SE(3) alignment requires at least three associations")
    source_center = np.mean(source, axis=0)
    target_center = np.mean(target, axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    u_matrix, _, vt_matrix = np.linalg.svd(covariance)
    rotation = vt_matrix.T @ u_matrix.T
    if np.linalg.det(rotation) < 0.0:
        vt_matrix[-1, :] *= -1.0
        rotation = vt_matrix.T @ u_matrix.T
    translation = target_center - rotation @ source_center
    return rotation, translation


def error_statistics(errors: np.ndarray, prefix: str = "") -> Dict[str, float]:
    errors = np.asarray(errors, dtype=np.float64)
    if errors.size == 0 or np.any(~np.isfinite(errors)):
        raise ValueError("error statistics require finite samples")
    return {
        f"{prefix}rmse_m": float(np.sqrt(np.mean(errors ** 2))),
        f"{prefix}mean_m": float(np.mean(errors)),
        f"{prefix}median_m": float(np.median(errors)),
        f"{prefix}p95_m": float(np.percentile(errors, 95)),
        f"{prefix}maximum_m": float(np.max(errors)),
    }


def interpolate_trajectory(samples: np.ndarray, query_times: np.ndarray,
                           maximum_gap_sec: float,
                           include_orientation: bool = False):
    """Interpolate one dense stream at query times and return an audit record."""
    if maximum_gap_sec <= 0.0:
        raise ValueError("maximum interpolation gap must be positive")
    times = samples[:, 0]
    query_times = np.asarray(query_times, dtype=np.float64)
    right = np.searchsorted(times, query_times, side="left")
    exact = ((right < times.size) &
             np.isclose(times[np.minimum(right, times.size - 1)], query_times,
                        rtol=0.0, atol=1e-9))
    inside = exact | ((right > 0) & (right < times.size))
    gaps = np.full(query_times.shape, np.inf, dtype=np.float64)
    interpolated = np.zeros((query_times.size, 7 if include_orientation else 3),
                            dtype=np.float64)

    exact_rows = np.flatnonzero(exact)
    if exact_rows.size:
        source_rows = right[exact_rows]
        interpolated[exact_rows, :3] = samples[source_rows, 1:4]
        if include_orientation:
            interpolated[exact_rows, 3:7] = samples[source_rows, 4:8]
        gaps[exact_rows] = 0.0

    between_rows = np.flatnonzero(inside & ~exact)
    if between_rows.size:
        left_indices = right[between_rows] - 1
        right_indices = right[between_rows]
        local_gaps = times[right_indices] - times[left_indices]
        gaps[between_rows] = local_gaps
        alpha = ((query_times[between_rows] - times[left_indices]) / local_gaps)
        interpolated[between_rows, :3] = (
            samples[left_indices, 1:4] + alpha[:, None] *
            (samples[right_indices, 1:4] - samples[left_indices, 1:4]))
        if include_orientation:
            interpolated[between_rows, 3:7] = quaternion_slerp(
                samples[left_indices, 4:8], samples[right_indices, 4:8], alpha)

    accepted = inside & (gaps <= maximum_gap_sec)
    audit = {
        "query_count": int(query_times.size),
        "associated_count": int(np.sum(accepted)),
        "exact_timestamp_count": int(np.sum(exact & accepted)),
        "interpolated_count": int(np.sum((~exact) & accepted)),
        "rejected_outside_span_count": int(np.sum(~inside)),
        "rejected_interpolation_gap_count": int(np.sum(inside & ~accepted)),
        "maximum_allowed_gap_sec": float(maximum_gap_sec),
        "maximum_used_gap_sec": float(np.max(gaps[accepted])) if np.any(accepted) else None,
    }
    return interpolated[accepted], accepted, audit


def position_rpe(times: np.ndarray, estimated: np.ndarray, reference: np.ndarray,
                 delta_sec: Optional[float] = None,
                 tolerance_sec: Optional[float] = None) -> Dict[str, object]:
    """Translation RPE on already aligned, timestamp-associated positions."""
    times = np.asarray(times, dtype=np.float64)
    if times.size < 2:
        raise ValueError("RPE requires at least two associations")
    if delta_sec is None:
        first = np.arange(times.size - 1)
        second = first + 1
        pairing = "adjacent_associated_samples"
    else:
        if delta_sec <= 0.0:
            raise ValueError("RPE delta must be positive")
        tolerance = tolerance_sec if tolerance_sec is not None else 0.5 * delta_sec
        targets = times + delta_sec
        candidate = np.searchsorted(times, targets, side="left")
        valid = candidate < times.size
        first = np.flatnonzero(valid)
        second = candidate[valid]
        valid_delta = (second > first) & (np.abs(
            (times[second] - times[first]) - delta_sec) <= tolerance)
        first, second = first[valid_delta], second[valid_delta]
        pairing = "fixed_time_delta"
    if first.size == 0:
        raise ValueError("no valid RPE pairs")
    residual = ((estimated[second] - estimated[first]) -
                (reference[second] - reference[first]))
    errors = np.linalg.norm(residual, axis=1)
    result: Dict[str, object] = {
        "pairing": pairing,
        "pairs": int(errors.size),
        "translation": error_statistics(errors),
    }
    if delta_sec is not None:
        result["delta_sec"] = float(delta_sec)
        result["maximum_delta_error_sec"] = float(tolerance)
    return result


def _fixed_delta_pairs(times: np.ndarray, delta_sec: float,
                       tolerance_sec: Optional[float]):
    tolerance = tolerance_sec if tolerance_sec is not None else 0.5 * delta_sec
    targets = times + delta_sec
    candidate = np.searchsorted(times, targets, side="left")
    valid = candidate < times.size
    first = np.flatnonzero(valid)
    second = candidate[valid]
    valid_delta = ((second > first) &
                   (np.abs((times[second] - times[first]) - delta_sec) <= tolerance))
    return first[valid_delta], second[valid_delta], tolerance


def pose_rpe(times: np.ndarray, estimated_positions: np.ndarray,
             estimated_quaternions: np.ndarray, reference_positions: np.ndarray,
             reference_quaternions: np.ndarray, delta_sec: float = 1.0,
             tolerance_sec: Optional[float] = None) -> Dict[str, object]:
    first, second, tolerance = _fixed_delta_pairs(
        np.asarray(times), delta_sec, tolerance_sec)
    if first.size == 0:
        raise ValueError("no valid pose RPE pairs")
    estimated_quaternions = normalize_quaternions(estimated_quaternions)
    reference_quaternions = normalize_quaternions(reference_quaternions)
    estimated_translation = inverse_rotate_vectors(
        estimated_quaternions[first],
        estimated_positions[second] - estimated_positions[first])
    reference_translation = inverse_rotate_vectors(
        reference_quaternions[first],
        reference_positions[second] - reference_positions[first])
    translation_errors = np.linalg.norm(
        estimated_translation - reference_translation, axis=1)
    estimated_relative = quaternion_multiply(
        quaternion_conjugate(estimated_quaternions[first]),
        estimated_quaternions[second])
    reference_relative = quaternion_multiply(
        quaternion_conjugate(reference_quaternions[first]),
        reference_quaternions[second])
    rotation_residual = normalize_quaternions(quaternion_multiply(
        quaternion_conjugate(reference_relative), estimated_relative))
    angles_deg = np.degrees(2.0 * np.arccos(
        np.clip(np.abs(rotation_residual[:, 3]), 0.0, 1.0)))
    rotation_stats = {
        "rmse_deg": float(np.sqrt(np.mean(angles_deg ** 2))),
        "mean_deg": float(np.mean(angles_deg)),
        "median_deg": float(np.median(angles_deg)),
        "p95_deg": float(np.percentile(angles_deg, 95)),
        "maximum_deg": float(np.max(angles_deg)),
    }
    return {
        "pairing": "fixed_time_delta",
        "delta_sec": float(delta_sec),
        "maximum_delta_error_sec": float(tolerance),
        "pairs": int(first.size),
        "translation": error_statistics(translation_errors),
        "rotation": rotation_stats,
    }


def sparse_position_metrics(truth: np.ndarray, estimate_positions: np.ndarray,
                            associated_mask: np.ndarray,
                            association_audit: Dict[str, object]) -> Dict[str, object]:
    truth_used = truth[associated_mask]
    if truth_used.shape[0] < 3:
        raise ValueError("fewer than three supported survey associations")
    rotation, translation = align_se3(estimate_positions, truth_used[:, 1:4])
    aligned = (rotation @ estimate_positions.T).T + translation
    errors = np.linalg.norm(aligned - truth_used[:, 1:4], axis=1)
    return {
        "metric_family": "sparse_survey_translation",
        "truth_interpolation": False,
        "estimate_interpolation": True,
        "rotation_ground_truth_available": False,
        "se3_aligned": True,
        "samples": int(errors.size),
        "overlap_start_sec": float(truth_used[0, 0]),
        "overlap_end_sec": float(truth_used[-1, 0]),
        "association": association_audit,
        "ate": error_statistics(errors, prefix="ate_"),
        "rpe": position_rpe(truth_used[:, 0], aligned, truth_used[:, 1:4]),
        "endpoint_error_m": float(errors[-1]),
        "alignment_rotation": rotation.tolist(),
        "alignment_translation_m": translation.tolist(),
    }


def evaluate_sparse_survey(truth: np.ndarray, estimate: np.ndarray,
                           maximum_estimate_gap_sec: float) -> Dict[str, object]:
    interpolated, mask, audit = interpolate_trajectory(
        estimate, truth[:, 0], maximum_estimate_gap_sec,
        include_orientation=False)
    return sparse_position_metrics(truth, interpolated, mask, audit)


def evaluate_position_reference(reference: np.ndarray, estimate: np.ndarray,
                                maximum_reference_gap_sec: float,
                                rpe_delta_sec: float = 1.0) -> Dict[str, object]:
    """Dense position ATE/RPE; reference interpolation is explicitly bounded."""
    reference_at_estimate, mask, audit = interpolate_trajectory(
        reference, estimate[:, 0], maximum_reference_gap_sec,
        include_orientation=False)
    estimate_used = estimate[mask]
    rotation, translation = align_se3(estimate_used[:, 1:4], reference_at_estimate)
    aligned = (rotation @ estimate_used[:, 1:4].T).T + translation
    errors = np.linalg.norm(aligned - reference_at_estimate, axis=1)
    result = {
        "metric_family": "dense_position",
        "reference_interpolation": True,
        "rotation_ground_truth_available": False,
        "se3_aligned": True,
        "samples": int(errors.size),
        "association": audit,
        "ate": error_statistics(errors, prefix="ate_"),
        "endpoint_error_m": float(errors[-1]),
        "alignment_rotation": rotation.tolist(),
        "alignment_translation_m": translation.tolist(),
    }
    try:
        result["rpe"] = position_rpe(
            estimate_used[:, 0], aligned, reference_at_estimate, rpe_delta_sec)
    except ValueError as error:
        result["rpe"] = {"available": False, "reason": str(error)}
    return result


def evaluate_dense_pose_reference(reference: np.ndarray, estimate: np.ndarray,
                                  maximum_reference_gap_sec: float,
                                  rpe_delta_sec: float = 1.0) -> Dict[str, object]:
    """Evaluate a dense 6-DoF TUM trajectory with bounded reference interpolation."""
    if reference.shape[1] < 8 or estimate.shape[1] < 8:
        raise ValueError("dense pose evaluation requires timestamp xyz xyzw")
    reference_pose, mask, audit = interpolate_trajectory(
        reference, estimate[:, 0], maximum_reference_gap_sec,
        include_orientation=True)
    estimate_used = estimate[mask]
    if estimate_used.shape[0] < 3:
        raise ValueError("fewer than three bounded pose associations")
    rotation, translation = align_se3(
        estimate_used[:, 1:4], reference_pose[:, :3])
    aligned_positions = (rotation @ estimate_used[:, 1:4].T).T + translation
    alignment_quaternion = rotation_matrix_to_quaternion(rotation)
    aligned_quaternions = quaternion_multiply(
        np.repeat(alignment_quaternion[None, :], estimate_used.shape[0], axis=0),
        normalize_quaternions(estimate_used[:, 4:8]))
    position_errors = np.linalg.norm(aligned_positions - reference_pose[:, :3], axis=1)
    orientation_residual = normalize_quaternions(quaternion_multiply(
        quaternion_conjugate(reference_pose[:, 3:7]), aligned_quaternions))
    orientation_errors_deg = np.degrees(2.0 * np.arccos(
        np.clip(np.abs(orientation_residual[:, 3]), 0.0, 1.0)))
    try:
        rpe = pose_rpe(
            estimate_used[:, 0], aligned_positions, aligned_quaternions,
            reference_pose[:, :3], reference_pose[:, 3:7], rpe_delta_sec)
    except ValueError as error:
        rpe = {"available": False, "reason": str(error)}
    return {
        "metric_family": "dense_6dof",
        "reference_interpolation": True,
        "rotation_ground_truth_available": True,
        "se3_aligned": True,
        "samples": int(position_errors.size),
        "association": audit,
        "ate": error_statistics(position_errors, prefix="ate_"),
        "rotation_ate": {
            "rmse_deg": float(np.sqrt(np.mean(orientation_errors_deg ** 2))),
            "mean_deg": float(np.mean(orientation_errors_deg)),
            "median_deg": float(np.median(orientation_errors_deg)),
            "p95_deg": float(np.percentile(orientation_errors_deg, 95)),
            "maximum_deg": float(np.max(orientation_errors_deg)),
        },
        "rpe": rpe,
        "endpoint_error_m": float(position_errors[-1]),
        "alignment_rotation": rotation.tolist(),
        "alignment_translation_m": translation.tolist(),
    }
