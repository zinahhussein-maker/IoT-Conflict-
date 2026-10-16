"""far_iot.metrics -- reconstructed, with non-conflicting-rule execution tracked."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict
import numpy as np


@dataclass
class RunLog:
    safety_violations: List[bool] = field(default_factory=list)
    resolution_correct: List[bool] = field(default_factory=list)
    override_events: List[str] = field(default_factory=list)
    shield_triggered: List[bool] = field(default_factory=list)
    nonconflicting_applied: List[int] = field(default_factory=list)  # NEW: count per tick
    nonconflicting_dropped: List[int] = field(default_factory=list)  # v2: gated-out non-conflicting rules per tick
    violated_names: List[tuple] = field(default_factory=list)        # v2: names of violated invariants per tick
    conflicts_seen: int = 0                                          # v2: number of conflicting pairs
    conflicts_deferred: int = 0                                      # v2: pairs with no admissible candidate


def safety_violation_rate(log: RunLog) -> float:
    if not log.safety_violations:
        return 0.0
    return float(np.mean(log.safety_violations))


def resolution_regret(log: RunLog) -> float:
    if not log.resolution_correct:
        return 0.0
    return 1.0 - float(np.mean(log.resolution_correct))


def fairness_override_rates(log: RunLog, owners: List[str]) -> Dict[str, float]:
    total = max(len(log.override_events), 1)
    counts = {o: 0 for o in owners}
    for o in log.override_events:
        counts[o] = counts.get(o, 0) + 1
    return {o: counts.get(o, 0) / total for o in owners}


def per_invariant_svr(log: RunLog, name: str) -> float:
    """v2: fraction of ticks at whose end invariant `name` is violated."""
    if not log.violated_names:
        return 0.0
    return float(np.mean([name in v for v in log.violated_names]))


def summarize(log: RunLog, owners: List[str]) -> Dict[str, float]:
    fr = fairness_override_rates(log, owners)
    return {
        "SVR_door": per_invariant_svr(log, "INV_away_mode_door_locked"),
        "SVR_humid": per_invariant_svr(log, "INV_ac_humidifier_conflict"),
        "SVR_fan_heater": per_invariant_svr(log, "INV_no_fan_heater_conflict"),
        "gate_intervention_rate": float(np.mean(log.shield_triggered)) if log.shield_triggered else 0.0,
        "n_conflicts": log.conflicts_seen,
        "deferred_rate": log.conflicts_deferred / max(log.conflicts_seen, 1),
        "total_nonconflicting_dropped": int(sum(log.nonconflicting_dropped)) if log.nonconflicting_dropped else 0,
        "SVR": safety_violation_rate(log),
        "regret": resolution_regret(log),
        "max_owner_override_rate": max(fr.values()) if fr else 0.0,
        "n_steps": len(log.safety_violations),
        "total_nonconflicting_applied": int(sum(log.nonconflicting_applied)) if log.nonconflicting_applied else 0,
    }
