"""
far_iot.baselines
==================
Reference resolution policies. Adds `static_shield_resolve`, the
reviewer-requested safety-aware static-priority baseline: deterministic
priority ordering (no learning, no adaptivity at all) combined with the
SAME single-step Shield safety check used by farid_shield. This isolates
a question the paper had not previously tested directly: does the safety
gate alone -- with no adaptive policy underneath it -- already achieve
farid_shield's safety profile, or does the learned policy contribute
something beyond what gating a fixed rule provides?
"""
from __future__ import annotations
import numpy as np
from typing import List, Optional
from .rules import Rule, State
from .shield import ShieldGate


def static_priority_resolve(candidates: List[Rule]) -> Rule:
    return max(candidates, key=lambda r: r.priority)


def ahp_adaptive_resolve(candidates: List[Rule], sigma: State, context_weights: Optional[dict] = None) -> Rule:
    context_weights = context_weights or {}
    def score(r: Rule) -> float:
        boost = 0.0
        if sigma.get("away_mode_flag") and r.owner == "vendor_default":
            boost += 3.0
        boost += context_weights.get(r.owner, 0.0)
        return r.priority + boost
    return max(candidates, key=score)


def random_resolve(candidates: List[Rule], rng: np.random.Generator) -> Rule:
    idx = rng.integers(0, len(candidates))
    return candidates[idx]


def static_shield_resolve(shield: ShieldGate, sigma: State, candidates: List[Rule], strict_block: bool = False) -> Optional[Rule]:
    """Deterministic highest-priority proposal, passed through the same
    single-step Shield check farid_shield uses -- no bandit, no learning."""
    proposed = static_priority_resolve(candidates)
    return shield.gate(sigma, proposed, candidates, strict_block=strict_block)
