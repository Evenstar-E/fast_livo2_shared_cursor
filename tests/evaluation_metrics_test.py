#!/usr/bin/env python3
"""Regression tests for paper trajectory, safety, and resource metrics."""

import csv
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from evaluation_metrics import (  # noqa: E402
    evaluate_dense_pose_reference,
    evaluate_sparse_survey,
    quaternion_multiply,
    rotate_vectors,
)


def quaternion_yaw(angle):
    return np.column_stack((np.zeros_like(angle), np.zeros_like(angle),
                            np.sin(angle / 2.0), np.cos(angle / 2.0)))


# Sparse HILTI: output cardinality is bounded by real survey observations, not
# by the dense estimate.  A truth point outside the estimate span is rejected.
truth_times = np.array([0.05, 1.05, 2.05, 4.0])
truth_xyz = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.1],
                      [1.0, 1.0, 0.2], [2.0, 1.0, 0.3]])
truth = np.column_stack((truth_times, truth_xyz))
estimate_times = np.arange(0.05, 3.06, 0.1)
angle = math.radians(25.0)
rotation = np.array([[math.cos(angle), -math.sin(angle), 0.0],
                     [math.sin(angle), math.cos(angle), 0.0],
                     [0.0, 0.0, 1.0]])
translation = np.array([3.0, -2.0, 0.5])
reference_dense = np.column_stack([
    np.interp(estimate_times, truth_times[:3], truth_xyz[:3, axis])
    for axis in range(3)])
estimate_xyz = (rotation.T @ (reference_dense - translation).T).T
estimate = np.column_stack((estimate_times, estimate_xyz))
sparse = evaluate_sparse_survey(truth, estimate, 0.11)
assert sparse["truth_interpolation"] is False
assert sparse["samples"] == 3
assert sparse["association"]["query_count"] == 4
assert sparse["association"]["rejected_outside_span_count"] == 1
assert sparse["ate"]["ate_rmse_m"] < 1e-10


# Dense 6DoF: a global rigid transform must disappear from ATE and from both
# components of fixed-delta RPE.
times = np.arange(0.0, 4.01, 0.1)
estimate_position = np.column_stack((times, np.sin(times), 0.1 * times ** 2))
estimate_quaternion = quaternion_yaw(0.1 * times)
alignment_q = quaternion_yaw(np.array([angle]))[0]
reference_position = (rotation @ estimate_position.T).T + translation
reference_quaternion = quaternion_multiply(
    np.repeat(alignment_q[None, :], times.size, axis=0), estimate_quaternion)
reference = np.column_stack((times, reference_position, reference_quaternion))
estimate_pose = np.column_stack((times, estimate_position, estimate_quaternion))
dense = evaluate_dense_pose_reference(reference, estimate_pose, 0.11, 1.0)
assert dense["ate"]["ate_rmse_m"] < 1e-10
assert dense["rotation_ate"]["rmse_deg"] < 1e-5
assert dense["rpe"]["translation"]["rmse_m"] < 1e-10
assert dense["rpe"]["rotation"]["rmse_deg"] < 1e-5


# The lever-arm primitive rotates a body-frame vector into the world frame.
quarter_turn = quaternion_yaw(np.array([math.pi / 2.0]))
rotated = rotate_vectors(quarter_turn, np.array([[1.0, 0.0, 0.0]]))
assert np.allclose(rotated, [[0.0, 1.0, 0.0]], atol=1e-12)


# Run-artifact extraction: detection timing, leakage, dwell, replay and HWM.
with tempfile.TemporaryDirectory() as temporary:
    run = Path(temporary)
    (run / "run.env").write_text(
        "dataset=HILTI22\nsequence=synthetic\ncondition=fault\n"
        "fault_mode=mixed\nfault_start_sec=10\nfault_end_sec=12\n",
        encoding="utf-8")
    (run / "resource_summary.env").write_text(
        "incumbent_peak_rss_kib=100\nincumbent_peak_hwm_kib=100\n"
        "candidate_peak_rss_kib=140\ncandidate_peak_hwm_kib=150\n"
        "total_peak_rss_kib=250\n", encoding="utf-8")
    (run / "active_odometry_summary.json").write_text(json.dumps({
        "strictly_increasing": True, "nonmonotonic_messages": 0,
        "final_generation_state": "candidate",
        "active_source_counts": {"incumbent": 10, "candidate": 3},
    }), encoding="utf-8")
    events = [
        {"event": "recovery_proposed", "fields": {"detection_sec": 9.0}},
        {"event": "recovery_proposed", "fields": {
            "detection_sec": 10.7, "point_onset_sec": 10.1,
            "onset_lower_bound_sec": 9.8, "anomaly_duration_sec": 0.6}},
        {"event": "recovery_proposed", "fields": {"detection_sec": 13.0}},
        {"event": "recovery_proposed", "fields": {"detection_sec": 23.0}},
        {"event": "replay_completed", "fields": {
            "replay_sensor_span_sec": 4.0, "replay_total_wall_sec": 2.0,
            "delta_peak_dirty_count": 42,
            "checkpoint_total_copy_wall_sec": 0.003}},
        {"event": "candidate_committed", "fields": {"relative_time_sec": 14.0}},
    ]
    (run / "shadow_progress_summary.json").write_text(
        json.dumps({"lifecycle_events": events}), encoding="utf-8")
    with (run / "active_odometry_callback_order.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("stamp_sec", "active_source"))
        writer.writeheader()
        writer.writerows([
            {"stamp_sec": 1.0, "active_source": "incumbent"},
            {"stamp_sec": 14.0, "active_source": "candidate"},
            {"stamp_sec": 15.0, "active_source": "candidate"},
            {"stamp_sec": 16.0, "active_source": "candidate"},
        ])
    output = run / "metrics.json"
    subprocess.run([
        sys.executable, str(REPO / "scripts" / "evaluate_run_metrics.py"),
        str(run), "--output", str(output)], check=True,
        stdout=subprocess.DEVNULL)
    measured = json.loads(output.read_text(encoding="utf-8"))
    assert abs(measured["detection"]["first_post_fault_detection_delay_sec"] - 0.7) < 1e-12
    assert measured["detection"]["pre_fault_proposal_count"] == 1
    assert measured["detection"]["in_fault_proposal_count"] == 1
    assert measured["detection"]["bounded_post_fault_proposal_count"] == 1
    assert measured["detection"]["late_detection_proposal_count"] == 1
    assert measured["detection"]["first_fault_relevant_detection_class"] == "in_fault"
    assert measured["detection"]["proposals"][1]["onset_lower_bound_covers_fault_start"]
    assert measured["safety"]["candidate_leakage_on_rejection"] is False
    assert measured["safety"]["post_switch_dwell"]["post_switch_dwell_sec"] == 2.0
    assert measured["resources"]["dirty_voxel_count"] == 42.0
    assert measured["resources"]["catchup_rate"] == 2.0
    assert measured["resources"]["candidate_peak_increment_over_incumbent_fraction"] == 0.5

print("EVALUATION_METRICS_TEST_OK")
