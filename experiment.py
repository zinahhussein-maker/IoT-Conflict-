"""
far_iot.experiment
===================
Reconstructed main simulation loop, now implementing the two central
fixes requested to strengthen experimental validation:

1. Non-conflicting triggered rules are now actually applied to the
   simulated state every tick (previously only conflict-resolution
   winners were ever applied; all non-conflicting rule activity was
   silently dropped -- Section IX-C's L9 gap). This is controlled by
   `apply_nonconflicting`, default True; pass False to reproduce the
   OLD (narrower) behavior for direct before/after comparison.
2. A new `static_shield` method (deterministic priority + Shield gate,
   no learning) is available alongside the existing methods, directly
   answering the reviewer's request for a safety-aware static-priority
   baseline.
"""
from __future__ import annotations
import time
import numpy as np
from typing import List, Dict, Optional

from .rules import Rule, State, conflict_pairs, non_conflicting_triggered
from .environment import (build_rule_corpus, build_invariants, initial_state,
                           EventStream, OWNERS, env_auto_transitions, EVENTS)
from .policy import LinUCBPolicy, build_context
from .shield import ShieldGate, VerifyGate
from .baselines import static_priority_resolve, ahp_adaptive_resolve, random_resolve, static_shield_resolve
from .metrics import RunLog, summarize

CONTEXT_DIM = 3 + len(OWNERS) + 2


def ground_truth_winner(rho_i: Rule, rho_j: Rule) -> Rule:
    if rho_i.priority != rho_j.priority:
        return rho_i if rho_i.priority > rho_j.priority else rho_j
    if rho_i.owner == "vendor_default":
        return rho_j
    if rho_j.owner == "vendor_default":
        return rho_i
    return rho_i


def apply_environment_event(sigma: State, event: str, rng: np.random.Generator) -> State:
    s = dict(sigma)
    if event == "away_mode":
        s["away_mode_flag"] = True
    elif event == "presence_home":
        s["away_mode_flag"] = False
    elif event == "humidity_high":
        s["humidity_high_flag"] = True
    elif event == "temp_low":
        s["humidity_high_flag"] = False
    return s


def run_single(method: str, seed: int, T: int = 500, n_rules: int = 60,
               drift_rate: float = 0.0, n_invariants: Optional[int] = None,
               verify_horizon: int = 5, strict_block: bool = False,
               apply_nonconflicting: bool = True,
               gate_nonconflicting: bool = True) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    rules = build_rule_corpus(seed=seed, n_rules=n_rules)
    all_invariants = build_invariants()
    invariants = all_invariants if n_invariants is None else all_invariants[:n_invariants]
    sigma = initial_state()
    stream = EventStream(EVENTS, seed=seed, drift_rate=drift_rate)

    shield = ShieldGate(invariants)
    verify_gate = VerifyGate(invariants, bounded_horizon=verify_horizon, env_transitions=env_auto_transitions())
    policy = LinUCBPolicy(dim=CONTEXT_DIM, alpha=0.8, seed=seed)

    log = RunLog()
    blocked_ticks = 0

    for t in range(T):
        event = stream.sample(t)
        sigma = apply_environment_event(sigma, event, rng)

        cpairs = conflict_pairs(rules, event, sigma)
        conflicting_ids = set()
        for (r1, r2) in cpairs:
            conflicting_ids.add(r1.id); conflicting_ids.add(r2.id)

        # --- NEW: apply all non-conflicting triggered rules directly ---
        # v2: non-conflicting rules pass through the SAME gate as resolutions
        # (Event-B: ApplyRule is guarded by the Shield in M1 and by Verify in
        # M2/M3; a rule failing the guard is discarded by DropRule).
        n_applied = 0; n_dropped = 0
        if apply_nonconflicting:
            for r in non_conflicting_triggered(rules, event, sigma, conflicting_ids):
                if gate_nonconflicting and method in ("farid_shield", "static_shield"):
                    admissible = shield.check(sigma, r.action)
                elif gate_nonconflicting and method == "farid_verify":
                    admissible = verify_gate._reachable_safe(sigma, r.action)
                else:
                    admissible = True
                if admissible:
                    sigma = r.apply(sigma); n_applied += 1
                else:
                    n_dropped += 1
        log.nonconflicting_applied.append(n_applied)
        log.nonconflicting_dropped.append(n_dropped)

        for (rho_i, rho_j) in cpairs:
            gt = ground_truth_winner(rho_i, rho_j)
            log.conflicts_seen += 1

            if method == "static":
                winner = static_priority_resolve([rho_i, rho_j]); shield_fired = False
            elif method == "ahp":
                winner = ahp_adaptive_resolve([rho_i, rho_j], sigma); shield_fired = False
            elif method == "random":
                winner = random_resolve([rho_i, rho_j], rng); shield_fired = False
            elif method == "static_shield":
                winner = static_shield_resolve(shield, sigma, [rho_i, rho_j], strict_block=strict_block)
                shield_fired = True
                if winner is None:
                    blocked_ticks += 1
                    continue
            elif method in ("farid_shield", "farid_verify", "farid_nogate"):
                ctx_i = build_context(rho_i, rho_j, sigma, OWNERS)
                ctx_j = build_context(rho_j, rho_i, sigma, OWNERS)
                chosen_id = policy.select([(rho_i.id, ctx_i), (rho_j.id, ctx_j)])
                proposed = rho_i if chosen_id == rho_i.id else rho_j

                if method == "farid_nogate":
                    winner = proposed; shield_fired = False
                elif method == "farid_shield":
                    winner = shield.gate(sigma, proposed, [rho_i, rho_j], strict_block=strict_block)
                    if winner is None:
                        blocked_ticks += 1; log.conflicts_deferred += 1
                        reward = -1.0
                        policy.update(proposed.id, ctx_i if proposed.id == rho_i.id else ctx_j, reward)
                        continue
                    shield_fired = (winner.id != proposed.id)
                else:  # farid_verify -- v2: mirrors FARIoT_M3 exactly
                    other = rho_j if proposed.id == rho_i.id else rho_i
                    if verify_gate._reachable_safe(sigma, proposed.action):
                        winner = proposed; shield_fired = False          # ResolveByPolicy
                    elif verify_gate._reachable_safe(sigma, other.action):
                        winner = other; shield_fired = True              # FallbackResolve
                    else:                                                 # DeferConflict
                        blocked_ticks += 1; log.conflicts_deferred += 1
                        log.shield_triggered.append(True)
                        policy.update(proposed.id, ctx_i if proposed.id == rho_i.id else ctx_j, -1.0)
                        continue

                reward = 1.0 if winner.id == gt.id else -1.0
                ctx_w = ctx_i if winner.id == rho_i.id else ctx_j
                policy.update(winner.id, ctx_w, reward)
            else:
                raise ValueError(f"unknown method {method}")

            if winner is None:
                continue
            sigma = winner.apply(sigma)
            loser = rho_j if winner.id == rho_i.id else rho_i
            log.resolution_correct.append(winner.id == gt.id)
            log.override_events.append(loser.owner)
            if method in ("farid_shield", "farid_verify", "farid_nogate", "static_shield"):
                log.shield_triggered.append(shield_fired)

        bad = tuple(inv.name for inv in invariants if not inv.holds(sigma))
        log.violated_names.append(bad)
        log.safety_violations.append(len(bad) > 0)

    result = summarize(log, OWNERS)
    result["blocked_ticks"] = blocked_ticks
    result["blocked_rate"] = blocked_ticks / max(T, 1)
    return result
