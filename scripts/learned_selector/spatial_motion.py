"""Causal, metric spatial/directional features; no model, labels, or map writes.

All geometric inputs and vectors must be in ONE frame. ``reference_rotation``
maps canonical (normally LiDAR) axes into that frame. A simultaneous global
SE(3) coordinate change therefore leaves the output unchanged. This is not
invariance to physically moving a sensor through a scene.
"""
from __future__ import annotations

import numpy as np

SCHEMA = "lio_spatial_motion_v1"
SECTORS = 12
VECTOR_KEYS = ("gyro_rad_s", "accel_m_s2", "gravity_m_s2", "velocity_m_s",
               "position_innovation_m", "rotation_innovation_rad", "velocity_innovation_m_s")
FEATURE_NAMES = (
    "log1p_input_points", "finite_nonzero_fraction", "log1p_sample_points",
    "range_q10_m", "range_q50_m", "range_q90_m", "range_mean_m", "range_std_m",
    "cov_eigen_max_m2", "cov_eigen_mid_m2", "cov_eigen_min_m2",
    "linearity", "planarity", "scattering", "sector_entropy", "sector_occupied_fraction", "sector_min_fraction",
) + tuple(f"sector_{s:02d}_{field}" for s in range(SECTORS) for field in ("fraction", "range_median_m")) + (
    "plane_support_fraction", "plane_residual_mean_m", "plane_residual_rmse_m",
    "plane_radial_jacobian_mean_m", "plane_radial_slope", "plane_offset_intercept_m",
    "plane_regression_rmse_m", "plane_sector_slope_mean", "plane_sector_slope_std",
    "plane_sector_sign_agreement",
) + tuple(f"{key}_{axis}" for key in VECTOR_KEYS for axis in "xyz") + (
    "gyro_rotation_innovation_cosine", "accel_gravity_cosine", "velocity_position_innovation_cosine",
    "velocity_velocity_innovation_cosine",
)


def deterministic_indices(count: int, maximum: int) -> np.ndarray:
    """Evenly spaced raw-message indices: repeatable, bounded, no random state.

    Input order is part of provenance. With subsampling this does NOT promise
    permutation invariance; do not reorder points between paired conditions.
    """
    if not isinstance(maximum, (int, np.integer)) or maximum < 4:
        raise ValueError("max_points must be an integer >= 4")
    if count <= maximum:
        return np.arange(count, dtype=np.int64)
    return (np.arange(maximum, dtype=np.int64) * count // maximum)


def _matrix(value, shape, name):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return result


def _regression(j, residual):
    if len(j) < 4:
        return None
    centered = j - np.mean(j)
    denominator = float(centered @ centered)
    if denominator <= 1e-12:
        return None
    slope = float(centered @ (residual - np.mean(residual)) / denominator)
    intercept = float(np.mean(residual) - slope * np.mean(j))
    rmse = float(np.sqrt(np.mean((residual - slope * j - intercept) ** 2)))
    return slope, intercept, rmse


def extract_features(points_xyz, *, sensor_origin=None, reference_rotation=None,
                     plane_normals=None, plane_offsets=None, matched_mask=None,
                     vectors=None, max_points=4096):
    """Return fixed names, finite values, missingness masks, and provenance counts.

    Plane convention: n dot p + d = 0; arrays align with EVERY input point.
    Optional vectors are actual simultaneous measurements/state innovations,
    not norms expanded into invented xyz. Nonfinite optional rows are missing.
    Plane summaries describe supplied correspondences, not independent truth.
    The caller owns acquisition timestamps and may use only arrived data.
    """
    p = np.asarray(points_xyz, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError("points_xyz must have shape N x 3")
    origin = _matrix(np.zeros(3) if sensor_origin is None else sensor_origin, (3,), "sensor_origin")
    rotation = _matrix(np.eye(3) if reference_rotation is None else reference_rotation, (3, 3), "reference_rotation")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-8):
        raise ValueError("reference_rotation must be a proper rotation")
    indices = deterministic_indices(len(p), max_points)
    radial_all = p - origin
    finite = np.isfinite(radial_all).all(axis=1)
    valid = finite & (np.linalg.norm(np.where(finite[:, None], radial_all, 0), axis=1) > 1e-9)
    indices = indices[valid[indices]]
    canonical = radial_all[indices] @ rotation
    values = {name: 0.0 for name in FEATURE_NAMES}
    mask = {name: 0 for name in FEATURE_NAMES}

    def put(name, value):
        if np.isfinite(value):
            values[name] = float(value)
            mask[name] = 1

    put("log1p_input_points", np.log1p(len(p)))
    put("log1p_sample_points", np.log1p(len(indices)))
    if len(p):
        put("finite_nonzero_fraction", np.mean(valid))
    sector = np.zeros(0, dtype=np.int64)
    if len(indices):
        radii = np.linalg.norm(canonical, axis=1)
        for name, value in zip(("range_q10_m", "range_q50_m", "range_q90_m"), np.quantile(radii, [0.1, 0.5, 0.9])):
            put(name, value)
        put("range_mean_m", np.mean(radii)); put("range_std_m", np.std(radii))
        angles = np.mod(np.arctan2(canonical[:, 1], canonical[:, 0]), 2 * np.pi)
        sector = np.minimum((angles * SECTORS / (2 * np.pi)).astype(np.int64), SECTORS - 1)
        counts = np.bincount(sector, minlength=SECTORS)
        fractions = counts / len(indices)
        nonzero = fractions[fractions > 0]
        put("sector_entropy", -np.sum(nonzero * np.log(nonzero)) / np.log(SECTORS))
        put("sector_occupied_fraction", np.mean(counts > 0)); put("sector_min_fraction", np.min(fractions))
        for s in range(SECTORS):
            put(f"sector_{s:02d}_fraction", fractions[s])
            if counts[s]:
                put(f"sector_{s:02d}_range_median_m", np.median(radii[sector == s]))
        if len(indices) >= 4:
            centered = canonical - np.mean(canonical, axis=0)
            eig = np.maximum(np.linalg.eigvalsh(centered.T @ centered / len(indices))[::-1], 0)
            for name, value in zip(("cov_eigen_max_m2", "cov_eigen_mid_m2", "cov_eigen_min_m2"), eig):
                put(name, value)
            if eig[0] > 1e-12:
                put("linearity", (eig[0] - eig[1]) / eig[0])
                put("planarity", (eig[1] - eig[2]) / eig[0])
                put("scattering", eig[2] / eig[0])

    if (plane_normals is None) != (plane_offsets is None):
        raise ValueError("plane_normals and plane_offsets must be provided together")
    if plane_normals is not None:
        normal = np.asarray(plane_normals, dtype=np.float64)
        offset = np.asarray(plane_offsets, dtype=np.float64)
        if normal.shape != p.shape or offset.shape != (len(p),):
            raise ValueError("plane arrays must align with original input points")
        supplied = np.ones(len(p), dtype=bool) if matched_mask is None else np.asarray(matched_mask, dtype=bool)
        if supplied.shape != (len(p),):
            raise ValueError("matched_mask must align with original input points")
        normal = normal[indices]; offset = offset[indices]
        norm = np.linalg.norm(normal, axis=1)
        good = supplied[indices] & np.isfinite(normal).all(axis=1) & np.isfinite(offset) & (norm > 1e-9)
        if len(indices):
            put("plane_support_fraction", np.mean(good))
        if np.any(good):
            n = normal[good] / norm[good, None]
            d = offset[good] / norm[good]
            j = np.sum(n * radial_all[indices[good]], axis=1)
            sign = np.where(j < 0, -1.0, 1.0)
            residual = (np.sum(n * p[indices[good]], axis=1) + d) * sign
            j = j * sign
            put("plane_residual_mean_m", np.mean(residual))
            put("plane_residual_rmse_m", np.sqrt(np.mean(residual ** 2)))
            put("plane_radial_jacobian_mean_m", np.mean(j))
            fit = _regression(j, residual)
            if fit is not None:
                for name, value in zip(("plane_radial_slope", "plane_offset_intercept_m", "plane_regression_rmse_m"), fit):
                    put(name, value)
            slopes = []
            for s in range(SECTORS):
                selected = sector[good] == s
                local_fit = _regression(j[selected], residual[selected])
                if local_fit is not None:
                    slopes.append(local_fit[0])
            if len(slopes) >= 2:
                slopes = np.asarray(slopes)
                put("plane_sector_slope_mean", np.mean(slopes))
                put("plane_sector_slope_std", np.std(slopes))
                # A zero/near-zero sign is explicitly neutral, not positive evidence.
                signs = np.sign(np.where(np.abs(slopes) < 1e-9, 0, slopes))
                put("plane_sector_sign_agreement", max(np.mean(signs > 0), np.mean(signs < 0)))
    elif matched_mask is not None:
        raise ValueError("matched_mask without supplied planes has no defined semantics")

    canonical_vectors = {}
    for key, vector in (vectors or {}).items():
        if key not in VECTOR_KEYS:
            raise ValueError(f"unknown vector: {key}")
        vector = np.asarray(vector, dtype=np.float64)
        if vector.shape != (3,):
            raise ValueError(f"{key} must have shape 3")
        if not np.isfinite(vector).all():
            continue
        v = vector @ rotation
        canonical_vectors[key] = v
        for axis, value in zip("xyz", v):
            put(f"{key}_{axis}", value)
    pairs = (("gyro_rotation_innovation_cosine", "gyro_rad_s", "rotation_innovation_rad"),
             ("accel_gravity_cosine", "accel_m_s2", "gravity_m_s2"),
             ("velocity_position_innovation_cosine", "velocity_m_s", "position_innovation_m"),
             ("velocity_velocity_innovation_cosine", "velocity_m_s", "velocity_innovation_m_s"))
    for name, a, b in pairs:
        if a in canonical_vectors and b in canonical_vectors:
            va, vb = canonical_vectors[a], canonical_vectors[b]
            denominator = np.linalg.norm(va) * np.linalg.norm(vb)
            if denominator > 1e-12:
                put(name, np.clip(va @ vb / denominator, -1, 1))
    return {"schema": SCHEMA, "names": list(FEATURE_NAMES),
            "values": [values[n] for n in FEATURE_NAMES], "mask": [mask[n] for n in FEATURE_NAMES],
            "metadata": {"input_count": len(p), "valid_count": int(np.sum(valid)), "sample_count": len(indices),
                         "max_points": int(max_points), "sampling": "even_raw_indices_before_valid_filter",
                         "frame": "canonical_axes_relative_to_sensor_origin", "causal": True,
                         "plane_correspondences_supplied": plane_normals is not None,
                         "geometry_is_quality_ground_truth": False, "map_action": "NONE"}}
