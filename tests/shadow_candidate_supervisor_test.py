#!/usr/bin/env python3

import json
import math
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from shadow_candidate_supervisor import atomic_write_json, evaluate  # noqa: E402


def write_run(directory, offsets, committed=True):
    directory.mkdir()
    with (directory / "trajectory.txt").open("w", encoding="utf-8") as stream:
        for index, offset in enumerate(offsets):
            timestamp = 1000.0 + index * 0.1
            stream.write(
                f"{timestamp:.6f} {index * 0.01 + offset:.9f} 0 0 0 0 0 1\n")
    (directory / "key_events.log").write_text(
        "Replay committed\n" if committed else "Candidate rejected\n",
        encoding="utf-8")


def main():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        incumbent = root / "incumbent"
        stable = root / "stable"
        divergent = root / "divergent"
        rejected = root / "rejected"
        sample_count = 120
        write_run(incumbent, [0.0] * sample_count)
        write_run(stable, [0.2] * sample_count)
        write_run(divergent, [0.2 + 0.01 * index for index in range(sample_count)])
        write_run(rejected, [0.0] * sample_count, committed=False)

        common = dict(
            probation_start_sec=1.0,
            probation_duration_sec=8.0,
            timestamp_tolerance_sec=0.001,
            maximum_boundary_position_divergence_m=0.5,
            maximum_boundary_rotation_divergence_deg=5.0,
            maximum_position_growth_m=0.5,
            maximum_rotation_growth_deg=5.0,
            minimum_samples=50)
        stable_result = evaluate(incumbent, stable, **common)
        assert stable_result["accepted"]
        assert stable_result["decision"] == "candidate"
        assert math.isclose(
            stable_result["metrics"]["boundary_position_divergence_m"], 0.2)

        divergent_result = evaluate(incumbent, divergent, **common)
        assert not divergent_result["accepted"]
        assert divergent_result["decision"] == "incumbent"

        rejected_result = evaluate(incumbent, rejected, **common)
        assert not rejected_result["accepted"]

        boundary_bad = root / "boundary_bad"
        write_run(boundary_bad, [0.8] * sample_count)
        boundary_bad_result = evaluate(incumbent, boundary_bad, **common)
        assert not boundary_bad_result["accepted"]

        manifest = root / "active_generation.json"
        atomic_write_json(manifest, divergent_result)
        loaded = json.loads(manifest.read_text(encoding="utf-8"))
        assert loaded["decision"] == "incumbent"

    print("SHADOW_CANDIDATE_SUPERVISOR_TEST_OK")


if __name__ == "__main__":
    main()
