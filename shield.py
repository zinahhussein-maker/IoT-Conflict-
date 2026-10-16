"""far_iot.shield -- Shield (Variant S) and Verify (Variant V) gates, reconstructed."""
from __future__ import annotations
from typing import List, Callable, Optional
from .rules import Rule, Invariant, State


class ShieldGate:
    def __init__(self, invariants: List[Invariant]):
        self.invariants = invariants

    def check(self, sigma: State, proposed_action: dict) -> bool:
        sigma_prime = dict(sigma); sigma_prime.update(proposed_action)
        return all(inv.holds(sigma_prime) for inv in self.invariants)

    def fallback(self, sigma: State, candidates: List[Rule], strict_block: bool = False) -> Optional[Rule]:
        if not candidates:
            return None
        ranked = sorted(candidates, key=lambda r: r.priority, reverse=True)
        for r in ranked:
            if self.check(sigma, r.action):
                return r
        if strict_block:
            return None
        return ranked[0]

    def gate(self, sigma: State, proposed_rule: Rule, candidates: List[Rule], strict_block: bool = False) -> Optional[Rule]:
        if self.check(sigma, proposed_rule.action):
            return proposed_rule
        return self.fallback(sigma, candidates, strict_block=strict_block)


class VerifyGate:
    def __init__(self, invariants: List[Invariant], bounded_horizon: int = 5,
                 env_transitions: Optional[List[Callable[[State], State]]] = None):
        self.invariants = invariants
        self.bounded_horizon = bounded_horizon
        self.env_transitions = env_transitions or []

    def _reachable_safe(self, sigma: State, candidate_action: dict) -> bool:
        s0 = dict(sigma); s0.update(candidate_action)
        if not all(inv.holds(s0) for inv in self.invariants):
            return False
        frontier = [s0]
        for _ in range(self.bounded_horizon):
            next_frontier = []
            for s in frontier:
                next_frontier.append(s)
                for trans in self.env_transitions:
                    s_prime = trans(s)
                    if not all(inv.holds(s_prime) for inv in self.invariants):
                        return False
                    next_frontier.append(s_prime)
            frontier = next_frontier[:50]
        return True

    def select_safe(self, sigma: State, candidates: List[Rule]) -> Optional[Rule]:
        ranked = sorted(candidates, key=lambda r: r.priority, reverse=True)
        for r in ranked:
            if self._reachable_safe(sigma, r.action):
                return r
        return None
