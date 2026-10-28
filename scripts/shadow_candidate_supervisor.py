#!/usr/bin/env python3
"""Select an isolated recovery candidate without mutating the incumbent run."""

import argparse
import json
import math
import os
import tempfile
from pathlib import Path


def load_trajectory(path):
    rows = []
    with open(path, "r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) != 8:
                raise ValueError(f"{path}:{line_number}: expected 8 columns")
            values = tuple(float(field) for field in fields)
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"{path}:{line_number}: non-finite value")
            rows.append(values)
    if not rows:
        raise ValueError(f"empty trajectory: {path}")
    if any(rows[index][0] <= rows[index - 1][0] for index in range(1, len(rows))):
        raise ValueError(f"trajectory timestamps are not strictly increasing: {path}")
    return rows


def match_trajectories(incumbent, candidate, tolerance_sec):
    pairs = []
    incumbent_index = 0
    candidate_index = 0
    while incumbent_index < len(incumbent) and candidate_index < len(candidate):
        delta = candidate[candidate_index][0] - incumbent[incumbent_index][0]
        if abs(delta) <= tolerance_sec:
            pairs.append((incumbent[incumbent_index], candidate[candidate_index]))
            incumbent_index += 1
            candidate_index += 1
        elif delta < 0.0:
            candidate_index += 1
        else:
            incumbent_index += 1
    return pairs


def position_divergence(first, second):
    return math.sqrt(sum((first[index] - second[index]) ** 2 for index in range(1, 4)))


def rotation_divergence_deg(first, second):
    dot = abs(sum(first[index] * second[index] for index in range(4, 8)))
    dot = max(0.0, min(1.0, dot))
    return math.degrees(2.0 * math.acos(dot))


def percentile(values, fraction):
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(math.ceil(fraction * len(ordered))) - 1)
    return ordered[max(0, index)]


def candidate_committed(candidate_dir):
    key_events = candidate_dir / "key_events.log"
    if not key_events.is_file():
        return False, "missing key_events.log"
    text = key_events.read_text(encoding="utf-8", errors="replace")
    if "Candidate rejected" in text:
        return False, "candidate transaction was rejected"
    if "Replay committed" not in text:
        return False, "candidate transaction did not commit"
    return True, "candidate transaction committed"


def evaluate(incumbent_dir, candidate_dir, probation_start_sec,
             probation_duration_sec, timestamp_tolerance_sec,
             maximum_boundary_position_divergence_m,
             maximum_boundary_rotation_divergence_deg,
             maximum_position_growth_m, maximum_rotation_growth_deg,
             minimum_samples):
    incumbent = load_trajectory(incumbent_dir / "trajectory.txt")
    candidate = load_trajectory(candidate_dir / "trajectory.txt")
    pairs = match_trajectories(incumbent, candidate, timestamp_tolerance_sec)
    origin_sec = max(incumbent[0][0], candidate[0][0])
    probation_end_sec = (probation_start_sec + probation_duration_sec
                         if probation_duration_sec > 0.0 else float("inf"))
    probation_pairs = [
        pair for pair in pairs
        if probation_start_sec <= pair[1][0] - origin_sec <= probation_end_sec
    ]

    committed, commit_reason = candidate_committed(candidate_dir)
    reasons = []
    if not committed:
        reasons.append(commit_reason)
    if len(probation_pairs) < minimum_samples:
        reasons.append(
            f"probation has {len(probation_pairs)} matched poses; need {minimum_samples}")

    metrics = {
        "matched_pose_count": len(pairs),
        "probation_pose_count": len(probation_pairs),
        "probation_start_sec": probation_start_sec,
        "probation_duration_sec": probation_duration_sec,
    }
    if probation_pairs:
        position = [position_divergence(*pair) for pair in probation_pairs]
        rotation = [rotation_divergence_deg(*pair) for pair in probation_pairs]
        position_growth = [max(0.0, value - position[0]) for value in position]
        rotation_growth = [max(0.0, value - rotation[0]) for value in rotation]
        metrics.update({
            "boundary_position_divergence_m": position[0],
            "boundary_rotation_divergence_deg": rotation[0],
            "maximum_position_divergence_m": max(position),
            "maximum_rotation_divergence_deg": max(rotation),
            "maximum_position_growth_m": max(position_growth),
            "maximum_rotation_growth_deg": max(rotation_growth),
            "p95_position_growth_m": percentile(position_growth, 0.95),
            "p95_rotation_growth_deg": percentile(rotation_growth, 0.95),
            "final_position_divergence_m": position[-1],
            "final_rotation_divergence_deg": rotation[-1],
        })
        if position[0] > maximum_boundary_position_divergence_m:
            reasons.append(
                f"boundary position divergence is {position[0]:.6f} m; "
                f"limit {maximum_boundary_position_divergence_m:.6f} m")
        if rotation[0] > maximum_boundary_rotation_divergence_deg:
            reasons.append(
                f"boundary rotation divergence is {rotation[0]:.6f} deg; "
                f"limit {maximum_boundary_rotation_divergence_deg:.6f} deg")
        if max(position_growth) > maximum_position_growth_m:
            reasons.append(
                f"position divergence grew {max(position_growth):.6f} m; "
                f"limit {maximum_position_growth_m:.6f} m")
        if max(rotation_growth) > maximum_rotation_growth_deg:
            reasons.append(
                f"rotation divergence grew {max(rotation_growth):.6f} deg; "
                f"limit {maximum_rotation_growth_deg:.6f} deg")

    accepted = not reasons
    return {
        "schema": 1,
        "decision": "candidate" if accepted else "incumbent",
        "accepted": accepted,
        "reasons": reasons or ["candidate committed and remained stable during probation"],
        "incumbent_directory": str(incumbent_dir.resolve()),
        "candidate_directory": str(candidate_dir.resolve()),
        "active_generation_directory": str(
            (candidate_dir if accepted else incumbent_dir).resolve()),
        "limits": {
            "timestamp_tolerance_sec": timestamp_tolerance_sec,
            "maximum_boundary_position_divergence_m":
                maximum_boundary_position_divergence_m,
            "maximum_boundary_rotation_divergence_deg":
                maximum_boundary_rotation_divergence_deg,
            "maximum_position_growth_m": maximum_position_growth_m,
            "maximum_rotation_growth_deg": maximum_rotation_growth_deg,
            "minimum_samples": minimum_samples,
        },
        "metrics": metrics,
    }


def atomic_write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--incumbent-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--probation-start-sec", type=float, required=True)
    parser.add_argument("--probation-duration-sec", type=float, default=20.0,
                        help="zero evaluates all remaining poses")
    parser.add_argument("--timestamp-tolerance-sec", type=float, default=0.02)
    parser.add_argument("--maximum-boundary-position-divergence-m", type=float,
                        default=1.0)
    parser.add_argument("--maximum-boundary-rotation-divergence-deg", type=float,
                        default=5.0)
    parser.add_argument("--maximum-position-growth-m", type=float, default=0.50)
    parser.add_argument("--maximum-rotation-growth-deg", type=float, default=5.0)
    parser.add_argument("--minimum-samples", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.probation_start_sec < 0.0 or args.probation_duration_sec < 0.0:
        parser.error("probation times must be non-negative")
    if (args.timestamp_tolerance_sec <= 0.0 or
            args.maximum_boundary_position_divergence_m <= 0.0 or
            args.maximum_boundary_rotation_divergence_deg <= 0.0 or
            args.maximum_position_growth_m <= 0.0):
        parser.error("tolerance and divergence limits must be positive")
    if args.maximum_rotation_growth_deg <= 0.0 or args.minimum_samples <= 0:
        parser.error("rotation growth limit and minimum samples must be positive")
    result = evaluate(
        args.incumbent_dir, args.candidate_dir, args.probation_start_sec,
        args.probation_duration_sec, args.timestamp_tolerance_sec,
        args.maximum_boundary_position_divergence_m,
        args.maximum_boundary_rotation_divergence_deg,
        args.maximum_position_growth_m, args.maximum_rotation_growth_deg,
        args.minimum_samples)
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
