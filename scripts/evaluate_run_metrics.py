#!/usr/bin/env python3
"""Extract detection, safety, lifecycle, and resource metrics from one run."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional


DEFAULT_RECOVERY_HORIZON_SEC = 10.0


def classify_detection(detection_sec: Optional[float], fault_start: Optional[float],
                       fault_end: Optional[float], recovery_horizon_sec: float,
                       condition: str) -> str:
    """Assign one mutually exclusive, protocol-level detection class."""
    if detection_sec is None:
        return "invalid_time"
    if condition != "fault" or fault_start is None or fault_end is None:
        return "clean_false"
    if detection_sec < fault_start:
        return "pre_fault"
    if detection_sec <= fault_end:
        return "in_fault"
    if detection_sec <= fault_end + recovery_horizon_sec:
        return "bounded_post_fault"
    return "late_detection"


def read_env(path: Path) -> Dict[str, str]:
    result: Dict[str, str] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                result[key] = value
    return result


def read_json(path: Path) -> Dict[str, object]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def finite_float(value) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def first_number(fields: Dict[str, object], *names: str) -> Optional[float]:
    for name in names:
        value = finite_float(fields.get(name))
        if value is not None:
            return value
    return None


def event_fields(event: Dict[str, object]) -> Dict[str, object]:
    fields = dict(event.get("fields", {}))
    if "relative_time_sec" not in fields:
        raw = str(event.get("status", ""))
        for token in raw.split(";"):
            if token.startswith("relative_time_sec="):
                fields["relative_time_sec"] = token.split("=", 1)[1]
    return fields


def post_switch_dwell(run: Path, final_state: str) -> Dict[str, object]:
    path = run / "active_odometry_callback_order.csv"
    if not path.is_file() or final_state != "candidate":
        return {"available": False,
                "reason": "no committed candidate" if final_state != "candidate" else "CSV missing"}
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    candidate = [row for row in rows if row.get("active_source") == "candidate"]
    if not candidate:
        return {"available": False, "reason": "candidate committed without canonical samples"}
    first = float(candidate[0]["stamp_sec"])
    last = float(candidate[-1]["stamp_sec"])
    return {"available": True, "samples": len(candidate),
            "first_candidate_stamp_sec": first,
            "last_candidate_stamp_sec": last,
            "post_switch_dwell_sec": last - first}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--recovery-horizon-sec", type=float,
        default=DEFAULT_RECOVERY_HORIZON_SEC,
        help="maximum eligible delay after labelled fault end (default: 10 s)")
    args = parser.parse_args()
    if args.recovery_horizon_sec < 0.0:
        raise ValueError("recovery horizon must be non-negative")
    run = args.run_directory.resolve()
    environment = read_env(run / "run.env")
    resources = read_env(run / "resource_summary.env")
    odometry = read_json(run / "active_odometry_summary.json")
    progress = read_json(run / "shadow_progress_summary.json")
    events: List[Dict[str, object]] = list(progress.get("lifecycle_events", []))
    raw_final_state = str(odometry.get("final_generation_state", "missing"))
    condition = environment.get("evaluation_condition", "")
    if not condition:
        condition = ("clean" if environment.get("fault_mode", "") == "none"
                     else "fault")
    configured_fault_start = finite_float(environment.get("fault_start_sec"))
    configured_fault_end = finite_float(environment.get("fault_end_sec"))
    fault_start = configured_fault_start if condition == "fault" else None
    fault_end = configured_fault_end if condition == "fault" else None

    proposals = []
    rejections = []
    commits = []
    replay_profiles = []
    checkpoint_profiles = []
    raw_proposals = []
    suppressed_proposals = []
    local_health_decisions = []
    map_decisions = []
    commit_decisions = []
    dry_run_eligible = []
    for event in events:
        name = event.get("event")
        fields = event_fields(event)
        relative_time = first_number(fields, "relative_time_sec")
        if name == "recovery_proposal_observed":
            raw_proposals.append(fields)
        if name == "recovery_proposal_suppressed":
            suppressed_proposals.append(fields)
        if name == "recovery_proposed":
            detection = first_number(fields, "detection_sec", "relative_time_sec")
            onset = first_number(fields, "point_onset_sec")
            lower = first_number(fields, "onset_lower_bound_sec")
            detection_class = classify_detection(
                detection, fault_start, fault_end, args.recovery_horizon_sec,
                condition)
            proposals.append({
                "detection_sec": detection,
                "detection_class": detection_class,
                "detection_delay_from_fault_start_sec": (
                    detection - fault_start if detection is not None and fault_start is not None else None),
                "point_onset_sec": onset,
                "point_onset_error_from_fault_start_sec": (
                    onset - fault_start if onset is not None and fault_start is not None else None),
                "onset_lower_bound_sec": lower,
                "onset_upper_bound_sec": first_number(
                    fields, "onset_upper_bound_sec"),
                "onset_confidence": first_number(fields, "onset_confidence"),
                "onset_interval_semantics": fields.get(
                    "onset_interval_semantics", "legacy_lower_bound_only"),
                "episode_id": first_number(fields, "episode_id"),
                "proposal_index_in_episode": first_number(
                    fields, "proposal_index_in_episode"),
                "merged_with_previous_proposal": str(
                    fields.get("merged_with_previous_proposal", "0")).lower()
                    in ("1", "true"),
                "episode_lower_bound_sec": first_number(
                    fields, "episode_lower_bound_sec"),
                "episode_upper_bound_sec": first_number(
                    fields, "episode_upper_bound_sec"),
                "residual_score": first_number(fields, "residual_score"),
                "match_score": first_number(fields, "match_score"),
                "innovation_score": first_number(fields, "innovation_score"),
                "fused_score": first_number(fields, "fused_score"),
                "onset_lower_bound_lead_from_fault_start_sec": (
                    fault_start - lower if lower is not None and fault_start is not None else None),
                "onset_lower_bound_covers_fault_start": (
                    lower <= fault_start <= detection
                    if lower is not None and fault_start is not None and detection is not None else None),
                "anomaly_duration_sec": first_number(fields, "anomaly_duration_sec"),
            })
        elif name == "candidate_rejected":
            rejections.append({"relative_time_sec": relative_time,
                               "reason": fields.get("detail", "unspecified")})
        elif name == "candidate_committed":
            commits.append({"relative_time_sec": relative_time,
                            "commit_delay_from_fault_end_sec": (
                                relative_time - fault_end
                                if relative_time is not None and fault_end is not None else None),
                            "detail": fields.get("detail", "")})
        elif name in ("replay_started", "replay_completed",
                      "replay_profile_finalized", "replay_failed_profile",
                      "candidate_validation_rejected_profile"):
            replay_profiles.append({"event": name, **fields})
        if name == "candidate_local_health_decision":
            local_health_decisions.append(fields)
        elif name == "candidate_map_decision":
            map_decisions.append(fields)
        elif name == "candidate_commit_decision":
            commit_decisions.append(fields)
        elif name == "candidate_commit_dry_run_eligible":
            dry_run_eligible.append(fields)
        if "checkpoint_total_copy_wall_sec" in fields:
            checkpoint_profiles.append({"event": name, **fields})

    latest_replay = next((item for item in reversed(replay_profiles)
                          if item["event"] in
                          ("replay_profile_finalized", "replay_completed")),
                         replay_profiles[-1] if replay_profiles else {})
    replay_sensor_span = first_number(latest_replay, "replay_sensor_span_sec")
    replay_wall = first_number(latest_replay, "replay_total_wall_sec",
                               "replay_event_processing_wall_sec")
    incumbent_hwm = finite_float(resources.get("incumbent_peak_hwm_kib"))
    candidate_hwm = finite_float(resources.get("candidate_peak_hwm_kib"))
    resource_increment = (
        (candidate_hwm - incumbent_hwm) / incumbent_hwm
        if incumbent_hwm and candidate_hwm is not None else None)
    candidate_samples = int(odometry.get("active_source_counts", {}).get("candidate", 0))
    normalized_rejection = (
        raw_final_state == "aborted" and bool(rejections) and candidate_samples == 0)
    final_state = "incumbent" if normalized_rejection else raw_final_state
    leakage = final_state != "candidate" and candidate_samples > 0
    checkpoint_wall_values = [first_number(item, "checkpoint_total_copy_wall_sec")
                              for item in checkpoint_profiles]
    checkpoint_wall_values = [value for value in checkpoint_wall_values
                              if value is not None]
    pre_fault_proposals = [proposal for proposal in proposals
                           if (fault_start is not None and
                               proposal["detection_sec"] is not None and
                               proposal["detection_sec"] < fault_start)]
    post_fault_proposals = [proposal for proposal in proposals
                            if (fault_start is not None and
                                proposal["detection_sec"] is not None and
                                proposal["detection_sec"] >= fault_start)]
    in_fault_proposals = [proposal for proposal in proposals
                          if proposal["detection_class"] == "in_fault"]
    bounded_post_fault_proposals = [
        proposal for proposal in proposals
        if proposal["detection_class"] == "bounded_post_fault"]
    late_proposals = [proposal for proposal in proposals
                      if proposal["detection_class"] == "late_detection"]
    recovery_eligible_proposals = in_fault_proposals + bounded_post_fault_proposals
    recovery_eligible_proposals.sort(
        key=lambda item: item["detection_sec"] if item["detection_sec"] is not None
        else math.inf)

    result = {
        "run_directory": str(run),
        "dataset": environment.get("dataset", ""),
        "sequence": environment.get("sequence", ""),
        "condition": condition,
        "fault_mode": environment.get("fault_mode", ""),
        "fault_start_sec": fault_start,
        "fault_end_sec": fault_end,
        "configured_fault_start_sec": configured_fault_start,
        "configured_fault_end_sec": configured_fault_end,
        "detection": {
            "recovery_horizon_sec": args.recovery_horizon_sec,
            "recovery_horizon_end_sec": (
                fault_end + args.recovery_horizon_sec
                if fault_end is not None else None),
            "proposal_count": len(proposals),
            "pre_fault_proposal_count": len(pre_fault_proposals),
            "post_fault_proposal_count": len(post_fault_proposals),
            "in_fault_proposal_count": len(in_fault_proposals),
            "bounded_post_fault_proposal_count": len(bounded_post_fault_proposals),
            "late_detection_proposal_count": len(late_proposals),
            "clean_false_proposal_count": len(proposals) if condition == "clean" else 0,
            "proposals": proposals,
            "first_proposal_class": (
                proposals[0]["detection_class"] if proposals else "no_detection"),
            "first_fault_relevant_detection_class": (
                post_fault_proposals[0]["detection_class"]
                if post_fault_proposals else "missed"),
            "first_post_fault_detection_delay_sec": (
                post_fault_proposals[0]["detection_delay_from_fault_start_sec"]
                if post_fault_proposals else None),
            "first_recovery_eligible_detection_delay_sec": (
                recovery_eligible_proposals[0]["detection_delay_from_fault_start_sec"]
                if recovery_eligible_proposals else None),
        },
        "safety": {
            "final_generation": final_state,
            "raw_final_generation": raw_final_state,
            "final_generation_normalized_from_rejection": normalized_rejection,
            "candidate_rejected": bool(rejections),
            "rejection_count": len(rejections),
            "latest_rejection_reason": (
                rejections[-1]["reason"] if rejections else None),
            "rejections": rejections,
            "canonical_monotonicity": bool(odometry.get("strictly_increasing", False)),
            "canonical_nonmonotonic_messages": odometry.get("nonmonotonic_messages"),
            "candidate_canonical_samples": candidate_samples,
            "candidate_leakage_on_rejection": leakage,
            "post_switch_dwell": post_switch_dwell(run, final_state),
            "commits": commits,
        },
        "resources": {
            "checkpoint_wall_sec_maximum": (
                max(checkpoint_wall_values) if checkpoint_wall_values else None),
            "checkpoint_wall_sec_latest": (
                checkpoint_wall_values[-1] if checkpoint_wall_values else None),
            "dirty_voxel_count": first_number(latest_replay, "delta_peak_dirty_count"),
            "dirty_visual_landmark_count": first_number(
                latest_replay, "visual_delta_before_image_count"),
            "replay_wall_sec": replay_wall,
            "replay_sensor_span_sec": replay_sensor_span,
            "catchup_rate": (
                replay_sensor_span / replay_wall
                if replay_sensor_span is not None and replay_wall and replay_wall > 0 else None),
            "incumbent_peak_rss_kib": finite_float(resources.get("incumbent_peak_rss_kib")),
            "incumbent_peak_hwm_kib": incumbent_hwm,
            "candidate_peak_rss_kib": finite_float(resources.get("candidate_peak_rss_kib")),
            "candidate_peak_hwm_kib": candidate_hwm,
            "candidate_peak_increment_over_incumbent_fraction": resource_increment,
            "total_peak_rss_kib": finite_float(resources.get("total_peak_rss_kib")),
        },
        "p2_commit_policy": {
            "raw_proposal_count": (len(raw_proposals)
                                   if raw_proposals else len(proposals)),
            "actionable_proposal_count": len(proposals),
            "suppressed_proposal_count": len(suppressed_proposals),
            "replay_attempt_count": sum(
                item["event"] == "replay_started" for item in replay_profiles),
            "latest_locally_healthy": (
                local_health_decisions[-1].get("locally_healthy")
                if local_health_decisions else None),
            "latest_map_acceptable": (
                map_decisions[-1].get("map_acceptable")
                if map_decisions else None),
            "latest_commit_eligible": (
                commit_decisions[-1].get("commit_eligible")
                if commit_decisions else None),
            "dry_run_eligible_count": len(dry_run_eligible),
            "stage_complete": bool(commit_decisions) and bool(map_decisions) and
                              bool(local_health_decisions) and any(
                                  item["event"] == "replay_completed"
                                  for item in replay_profiles),
        },
    }
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    output = args.output or run / "run_metrics.json"
    output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
