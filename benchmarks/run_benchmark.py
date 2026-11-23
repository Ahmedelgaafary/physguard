"""Synthetic benchmark: how many wrong numeric answers does physguard catch?

Each scenario has a ground-truth reference. We generate candidate answers with known
error types and measure (a) how many wrong answers the guard rejects and (b) how many
correct answers it wrongly rejects. SYNTHETIC: replace `candidates()` with real LLM
outputs (see README, "Next steps") before quoting results as model performance.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from physguard import LAWS, Claim, Guard  # noqa: E402
from physguard.units import parse_quantity as P  # noqa: E402

SCENARIOS = [
    ("cantilever_tip_deflection", {"F": "1 kN", "L": "2 m", "E": "200 GPa", "I": "8e-6 m**4"}),
    ("cantilever_tip_deflection", {"F": "500 N", "L": "1.5 m", "E": "70 GPa", "I": "2e-6 m**4"}),
    ("simply_supported_midspan_deflection", {"F": "10 kN", "L": "4 m", "E": "30 GPa", "I": "1e-4 m**4"}),
    ("ohms_law_current", {"V": "230 V", "R": "47 ohm"}),
    ("ohms_law_current", {"V": "5 V", "R": "330 ohm"}),
    ("kinetic_energy", {"m": "1200 kg", "v": "25 m/s"}),
    ("kinetic_energy", {"m": "0.145 kg", "v": "40 m/s"}),
    ("projectile_range", {"v": "30 m/s", "angle_deg": "30"}),
    ("ideal_gas_pressure", {"n": "2 mol", "T": "300 K", "V": "10 L"}),
    ("hydrostatic_pressure", {"rho": "1000 kg/m**3", "h": "25 m"}),
]


def candidates(ref):
    """(label, answer string, is_correct)"""
    u = ref.units
    v = ref.magnitude
    yield "correct", f"{v:.6g} {u}", True
    yield "correct_other_unit", f"{ref.to_base_units().magnitude:.6g} {ref.to_base_units().units}", True
    yield "prefix_x1000", f"{v * 1000:.6g} {u}", False
    yield "prefix_x1e-3", f"{v / 1000:.6g} {u}", False
    yield "off_by_15pct", f"{v * 1.15:.6g} {u}", False
    yield "off_by_2x", f"{v * 2:.6g} {u}", False
    yield "wrong_dimension", f"{v:.6g} kg", False


def main() -> None:
    guard = Guard(rtol=0.01)
    stats = {"wrong_total": 0, "wrong_caught": 0, "right_total": 0, "right_rejected": 0}
    by_type: dict[str, list[int]] = {}
    for law_name, inputs in SCENARIOS:
        ref = LAWS[law_name].func(**{k: P(v) for k, v in inputs.items()})
        for label, ans, correct in candidates(ref):
            rejected = not guard.verify(Claim(answer=ans, law=law_name, inputs=inputs)).ok
            by_type.setdefault(label, [0, 0])
            by_type[label][0] += 1
            by_type[label][1] += int(rejected)
            if correct:
                stats["right_total"] += 1
                stats["right_rejected"] += int(rejected)
            else:
                stats["wrong_total"] += 1
                stats["wrong_caught"] += int(rejected)

    print("| error type | cases | rejected |")
    print("|---|---|---|")
    for label, (n, r) in by_type.items():
        print(f"| {label} | {n} | {r} |")
    wc = stats["wrong_caught"] / stats["wrong_total"]
    fr = stats["right_rejected"] / stats["right_total"]
    print(f"\nWrong answers caught: {stats['wrong_caught']}/{stats['wrong_total']} ({wc:.0%})")
    print(f"Correct answers wrongly rejected: {stats['right_rejected']}/{stats['right_total']} ({fr:.0%})")
    print("Baseline (no guard) catches 0% of wrong answers.")


if __name__ == "__main__":
    main()
