"""
far_iot.rules
=============
Core ECA rule representation and the pairwise conflict predicate.
Reconstructed from the cumulative FAR-IoT simulation codebase.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Any, Optional
import itertools

State = Dict[str, Any]


@dataclass(frozen=True)
class Rule:
    id: str
    event: str
    cond: Callable[[State], bool]
    action: Dict[str, Any]
    priority: int
    owner: str
    conflict_class: Optional[str] = None

    def triggered(self, event: str, sigma: State) -> bool:
        return event == self.event and bool(self.cond(sigma))

    def apply(self, sigma: State) -> State:
        new_sigma = dict(sigma)
        new_sigma.update(self.action)
        return new_sigma


def conflicts(rho_i: Rule, rho_j: Rule, sigma: State) -> bool:
    if rho_i.id == rho_j.id:
        return False
    shared_vars = set(rho_i.action.keys()) & set(rho_j.action.keys())
    if not shared_vars:
        return False
    for var in shared_vars:
        if rho_i.action[var] != rho_j.action[var]:
            return True
    return False


def triggered_set(rules, event: str, sigma: State):
    return [r for r in rules if r.triggered(event, sigma)]


def conflict_pairs(rules, event: str, sigma: State):
    trig = triggered_set(rules, event, sigma)
    pairs = []
    for a, b in itertools.combinations(trig, 2):
        if conflicts(a, b, sigma):
            pairs.append((a, b))
    return pairs


def non_conflicting_triggered(rules, event: str, sigma: State, conflicting_ids: set):
    """
    NEW (Step: strengthen experimental validation). Returns triggered rules
    that are NOT part of any conflicting pair this tick -- i.e. rules whose
    actions can be applied directly, with no gate, because no other
    triggered rule this tick disagrees with them on a shared variable.
    Previously these rules were never applied to the simulated state at
    all (Section IX-C, L9 / L8 in earlier drafts); this function is what
    closes that gap.
    """
    trig = triggered_set(rules, event, sigma)
    return [r for r in trig if r.id not in conflicting_ids]


@dataclass
class Invariant:
    name: str
    check: Callable[[State], bool]
    description: str = ""

    def holds(self, sigma: State) -> bool:
        return bool(self.check(sigma))
