"""Causal, advisory-only action contract. No ROS, mapper or actuator side effects.

Predicted anomaly, identifiable cause and demonstrated action utility are distinct.
Receipt fields are checked here; their truth must be established by an independent
artifact auditor. A serialized Decision is never an actuator command or a public
map-commit permit.
"""
from dataclasses import asdict, dataclass
from enum import Enum
import math
import re
from typing import Optional, Tuple


class Status(str, Enum):
    UNKNOWN = "UNKNOWN"
    NORMAL = "NORMAL"
    ANOMALOUS = "ANOMALOUS"


class Action(str, Enum):
    NONE = "NONE"
    QUALITY_PROTECT = "QUALITY_PROTECT"
    SCALE_REPLAY = "SCALE_REPLAY"


def _digest(value):
    return bool(re.fullmatch(r"[0-9a-f]{64}", value or ""))


@dataclass(frozen=True)
class EvidenceFrame:
    event_id: str
    stamp_sec: float
    available_sec: float
    frame_id: str
    source_digest: str
    scale_probability: Optional[float] = None
    quality_probability: Optional[float] = None
    identifiable_scale: Optional[bool] = None
    independent_evidence_refs: Tuple[str, ...] = ()
    correction_scale: Optional[float] = None
    candidate_id: Optional[str] = None
    endpoint_digest: Optional[str] = None
    evidence_origin: str = "causal_measurements"
    quality_is_proxy: bool = True


@dataclass(frozen=True)
class ValidationEvidence:
    """Already audited, causal private-execution receipt for exactly one plan.

    This is not permission to use clean/fault labels or future trajectory truth
    online. Runtime validators must use only causal observables. Offline utility
    comparisons are reported separately and cannot be inserted as online receipts.
    """
    proposal_id: str
    event_id: str
    source_digest: str
    plan_id: str
    checkpoint_digest: str
    action: Action
    candidate_id: str
    endpoint_digest: str
    available_sec: float
    evaluated_through_sec: float
    private_execution_validated: bool = False
    clean_safe: bool = False
    map_noninferior: bool = False
    map_improved: bool = False
    trajectory_noninferior: bool = False
    coverage_noninferior: bool = False
    evidence_refs: Tuple[str, ...] = ()
    causal_only: bool = False


@dataclass(frozen=True)
class ControllerAdvisory:
    schema: str
    event_id: str
    frame_id: str
    source_digest: str
    issued_sec: float
    expires_sec: float
    request: str
    reason: str
    advisory_only: bool = True
    execution_authorized: bool = False

    def valid_at(self, now_sec):
        return math.isfinite(now_sec) and self.issued_sec <= now_sec < self.expires_sec


@dataclass(frozen=True)
class Decision:
    event_id: str
    proposal_id: Optional[str]
    scale_status: Status
    quality_status: Status
    summary: str
    proposed_action: Action
    validated_private_action: Action
    validation_reason: str
    plan_id: Optional[str]
    new_proposal: bool
    advisory: ControllerAdvisory
    public_commit_authorized: bool = False
    actuator_authorized: bool = False

    def to_dict(self):
        return asdict(self)


class SequentialActionPolicy:
    """Confirmation and rearm operate per head, preserving coexisting issues.

    0.5/3/3 are a frozen initial diagnostic convention, not calibrated safety
    thresholds. UNKNOWN breaks confirmation but never counts as healthy rearm.
    One open proposal persists through unknown/poor observations; a later scale
    cause can upgrade a quality proposal without losing its denominator record.
    """
    def __init__(self, threshold=0.5, confirm=3, rearm=3, ttl_sec=1.0,
                 max_input_age_sec=2.0, max_event_gap_sec=1.0):
        if not 0 < threshold < 1 or confirm < 1 or rearm < 1:
            raise ValueError("invalid confirmation configuration")
        if any(not math.isfinite(v) or v <= 0 for v in
               (ttl_sec, max_input_age_sec, max_event_gap_sec)):
            raise ValueError("invalid temporal bounds")
        self.threshold, self.confirm, self.rearm = threshold, confirm, rearm
        self.ttl_sec, self.max_input_age_sec = ttl_sec, max_input_age_sec
        self.max_event_gap_sec = max_event_gap_sec
        self.reset()

    def reset(self):
        self.last_stamp = None
        self.last_event = None
        self.counts = [[0, 0], [0, 0]]
        self.healthy_count = 0
        self.proposal_id = None
        self.proposals = 0

    def _head(self, score, index):
        if score is None:
            self.counts[index] = [0, 0]
            return Status.UNKNOWN
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("probability must be finite [0, 1] or None")
        abnormal = score >= self.threshold
        self.counts[index][int(abnormal)] += 1
        self.counts[index][int(not abnormal)] = 0
        if self.counts[index][int(abnormal)] < self.confirm:
            return Status.UNKNOWN
        return Status.ANOMALOUS if abnormal else Status.NORMAL

    def _validate(self, frame, action, receipt, now):
        if action == Action.NONE:
            return Action.NONE, "no_repair_proposed", None
        if receipt is None:
            return Action.NONE, "real_action_utility_unverified", None
        if frame.evidence_origin != "causal_measurements":
            return Action.NONE, "oracle_or_label_evidence_forbidden", None
        if (receipt.proposal_id != self.proposal_id or receipt.event_id != frame.event_id
                or receipt.source_digest != frame.source_digest or receipt.action != action
                or not frame.candidate_id or receipt.candidate_id != frame.candidate_id
                or not _digest(frame.endpoint_digest)
                or receipt.endpoint_digest != frame.endpoint_digest):
            return Action.NONE, "receipt_identity_mismatch", None
        if (not math.isfinite(receipt.available_sec)
                or not math.isfinite(receipt.evaluated_through_sec)
                or receipt.available_sec > now
                or receipt.evaluated_through_sec != frame.stamp_sec
                or receipt.available_sec < receipt.evaluated_through_sec
                or not receipt.causal_only):
            return Action.NONE, "noncausal_or_future_receipt", None
        if (not receipt.plan_id or not _digest(receipt.checkpoint_digest)
                or not receipt.evidence_refs or any(not x for x in receipt.evidence_refs)):
            return Action.NONE, "receipt_provenance_missing", None
        safe = (receipt.private_execution_validated and receipt.clean_safe
                and receipt.map_noninferior and receipt.trajectory_noninferior
                and receipt.coverage_noninferior)
        if not safe or (action == Action.SCALE_REPLAY and not receipt.map_improved):
            return Action.NONE, "joint_utility_not_passed", None
        return action, "validated_private_only_not_public_permission", receipt.plan_id

    def step(self, frame: EvidenceFrame, decision_time_sec: float,
             validation: Optional[ValidationEvidence] = None):
        now = decision_time_sec
        times = (frame.stamp_sec, frame.available_sec, now)
        if any(not math.isfinite(x) for x in times):
            raise ValueError("nonfinite timestamp")
        if frame.available_sec < frame.stamp_sec or now < frame.available_sec:
            raise ValueError("future or unavailable input")
        if not frame.event_id or not frame.frame_id or not _digest(frame.source_digest):
            raise ValueError("event/frame/SHA256 provenance required")
        if self.last_stamp is not None and (frame.stamp_sec <= self.last_stamp
                                             or frame.event_id == self.last_event):
            raise ValueError("events must be unique and strictly increasing")
        # Validate both scores before mutating state, making rejected inputs atomic.
        for score in (frame.scale_probability, frame.quality_probability):
            if score is not None and (not math.isfinite(score) or not 0 <= score <= 1):
                raise ValueError("invalid probability")
        if frame.identifiable_scale not in (None, True, False):
            raise ValueError("identifiability must be tri-state")
        stale = now - frame.stamp_sec > self.max_input_age_sec
        gap = self.last_stamp is not None and frame.stamp_sec-self.last_stamp > self.max_event_gap_sec
        if gap:
            self.counts = [[0, 0], [0, 0]]
            self.healthy_count = 0
        self.last_stamp, self.last_event = frame.stamp_sec, frame.event_id
        s = self._head(None if stale else frame.scale_probability, 0)
        q = self._head(None if stale else frame.quality_probability, 1)
        self.healthy_count = self.healthy_count + 1 if s == q == Status.NORMAL else 0
        if self.healthy_count >= self.rearm:
            self.proposal_id = None
        new = False
        if Status.ANOMALOUS in (s, q) and self.proposal_id is None:
            self.proposals += 1
            self.proposal_id = "proposal-%06d" % self.proposals
            new = True
        scale_identified = (s == Status.ANOMALOUS and frame.identifiable_scale is True
                            and bool(frame.independent_evidence_refs)
                            and all(bool(x) for x in frame.independent_evidence_refs))
        correction_valid = (frame.correction_scale is not None
                            and math.isfinite(frame.correction_scale)
                            and .92 <= frame.correction_scale <= 1.08
                            and frame.correction_scale != 1.0)
        if scale_identified and correction_valid:
            action = Action.SCALE_REPLAY
        elif q == Status.ANOMALOUS:
            action = Action.QUALITY_PROTECT
        else:
            action = Action.NONE
        if s == q == Status.NORMAL:
            summary = "PREDICTED_HEALTHY_OBSERVATION"
        elif scale_identified:
            summary = "SCALE_FAULT_WITH_QUALITY_LOSS" if q == Status.ANOMALOUS else "SCALE_FAULT_SUPPORTED"
        elif q == Status.ANOMALOUS:
            summary = "MATCHING_DIFFICULTY_SCALE_UNRESOLVED" if s != Status.NORMAL else "MATCHING_DIFFICULTY"
        else:
            summary = "UNKNOWN"
        authorized, reason, plan = self._validate(frame, action, validation, now)
        if stale:
            reason = "stale_input"
        request = "NONE" if summary == "PREDICTED_HEALTHY_OBSERVATION" else "REOBSERVE_REQUEST"
        advisory = ControllerAdvisory("lio-observation-advisory/v1", frame.event_id,
                                     frame.frame_id, frame.source_digest, now,
                                     now+self.ttl_sec, request, summary)
        return Decision(frame.event_id, self.proposal_id, s, q, summary, action,
                        authorized, reason, plan, new, advisory)
