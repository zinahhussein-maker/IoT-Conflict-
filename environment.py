"""
far_iot.environment
====================
Rule corpus, safety invariants, exogenous environment transitions, and
event stream. This version implements the reviewer-requested strengthening:
a SECOND, independently-designed multi-step scenario (humidity/AC trap),
built the same principled way as the original away-mode/door-lock trap --
randomized per seed, not hand-fixed -- so the central empirical claim no
longer rests on a single scenario template alone.
"""
from __future__ import annotations
import numpy as np
from typing import List, Dict
from .rules import Rule, Invariant, State

DEVICES = [
    "living_room_light", "bedroom_light", "hallway_light",
    "thermostat", "fan", "heater", "door_lock", "window_blind",
    "ac", "humidifier",
]
OWNERS = ["resident_A", "resident_B", "resident_C", "vendor_default"]

EVENTS = ["motion_detected", "temp_high", "temp_low", "door_opened",
          "time_evening", "time_morning", "humidity_high", "away_mode",
          "presence_home"]


def build_rule_corpus(seed: int = 0, n_rules: int = 60):
    rng = np.random.default_rng(seed)
    rules: List[Rule] = []

    device_domains: Dict[str, list] = {
        "living_room_light": ["ON", "OFF"], "bedroom_light": ["ON", "OFF"],
        "hallway_light": ["ON", "OFF"], "thermostat": [65, 68, 70, 72, 75, 78],
        "fan": ["ON", "OFF"], "heater": ["ON", "OFF"],
        "door_lock": ["LOCKED", "UNLOCKED"], "window_blind": ["OPEN", "CLOSED"],
        "ac": ["ON", "OFF"], "humidifier": ["ON", "OFF"],
    }

    rid = 0
    seed_conflicts = [
        ("temp_high", "fan", "resident_A", "ON", "vendor_default", "OFF", "oppositional", 5, 2),
        ("temp_high", "ac", "resident_B", "ON", "resident_C", "OFF", "oppositional", 4, 4),
        ("motion_detected", "living_room_light", "resident_A", "ON", "vendor_default", "OFF", "oppositional", 3, 1),
        ("time_evening", "thermostat", "resident_A", 72, "resident_B", 68, "numeric", 3, 3),
        ("humidity_high", "humidifier", "resident_C", "ON", "vendor_default", "OFF", "oppositional", 2, 6),
        ("temp_low", "heater", "resident_A", "ON", "resident_C", "OFF", "oppositional", 4, 3),
        ("presence_home", "window_blind", "resident_B", "OPEN", "vendor_default", "CLOSED", "shadow", 3, 2),
        ("time_morning", "bedroom_light", "resident_C", "ON", "resident_A", "OFF", "oppositional", 2, 2),
        ("door_opened", "hallway_light", "vendor_default", "ON", "resident_B", "OFF", "semantic", 1, 5),
    ]
    for (ev, dev, o1, v1, o2, v2, cls, p1, p2) in seed_conflicts:
        rules.append(Rule(id=f"r{rid}", event=ev, cond=(lambda s: True),
                           action={dev: v1}, priority=p1, owner=o1, conflict_class=cls)); rid += 1
        rules.append(Rule(id=f"r{rid}", event=ev, cond=(lambda s: True),
                           action={dev: v2}, priority=p2, owner=o2, conflict_class=cls)); rid += 1

    # --- Trap 1: door-unlock / away-mode (randomized per seed) ---
    trap1_owners = list(rng.choice(OWNERS, size=2, replace=False))
    trap1_prios = list(rng.integers(1, 9, size=2))
    rules.append(Rule(id=f"r{rid}", event="presence_home", cond=(lambda s: True),
                       action={"door_lock": "UNLOCKED"}, priority=int(trap1_prios[0]),
                       owner=trap1_owners[0], conflict_class="dependency")); rid += 1
    rules.append(Rule(id=f"r{rid}", event="presence_home", cond=(lambda s: True),
                       action={"door_lock": "LOCKED"}, priority=int(trap1_prios[1]),
                       owner=trap1_owners[1], conflict_class="dependency")); rid += 1

    # --- Trap 2 (NEW, independent design): ac-on / humidity-high ---
    # Mirrors Trap 1's structure (a device write that is safe now but
    # foreseeably unsafe once an exogenous transition fires later with no
    # re-decision in between) but exercises a DIFFERENT invariant (inv4:
    # ac/humidifier) via a DIFFERENT device, DIFFERENT triggering event,
    # and an independently-designed exogenous transition (humidity_high),
    # not a parameter reshuffle of Trap 1.
    trap2_owners = list(rng.choice(OWNERS, size=2, replace=False))
    trap2_prios = list(rng.integers(1, 9, size=2))
    rules.append(Rule(id=f"r{rid}", event="temp_high", cond=(lambda s: True),
                       action={"ac": "ON"}, priority=int(trap2_prios[0]),
                       owner=trap2_owners[0], conflict_class="dependency")); rid += 1
    rules.append(Rule(id=f"r{rid}", event="temp_high", cond=(lambda s: True),
                       action={"ac": "OFF"}, priority=int(trap2_prios[1]),
                       owner=trap2_owners[1], conflict_class="dependency")); rid += 1
    # An UNRELATED, NON-CONFLICTING rule that can independently turn the
    # humidifier on (used to test non-conflicting-rule execution, below).
    rules.append(Rule(id=f"r{rid}", event="temp_high", cond=(lambda s: True),
                       action={"humidifier": "ON"}, priority=1,
                       owner="resident_B", conflict_class=None)); rid += 1

    while len(rules) < n_rules:
        ev = str(rng.choice(EVENTS))
        dev = str(rng.choice(DEVICES))
        owner = str(rng.choice(OWNERS))
        val_options = device_domains[dev]
        val = val_options[int(rng.integers(0, len(val_options)))]
        prio = int(rng.integers(1, 6))
        rules.append(Rule(id=f"r{rid}", event=ev, cond=(lambda s: True),
                           action={dev: val}, priority=prio, owner=owner, conflict_class=None))
        rid += 1
    return rules


def build_invariants() -> List[Invariant]:
    return [
        Invariant("INV_no_fan_heater_conflict",
                  lambda s: not (s.get("fan") == "ON" and s.get("heater") == "ON"),
                  "fan and heater never simultaneously ON"),
        Invariant("INV_thermostat_bounds",
                  lambda s: 60 <= s.get("thermostat", 70) <= 80,
                  "thermostat within [60, 80]"),
        Invariant("INV_away_mode_door_locked",
                  lambda s: not (s.get("away_mode_flag") is True and s.get("door_lock") == "UNLOCKED"),
                  "door locked whenever away_mode is active"),
        Invariant("INV_ac_humidifier_conflict",
                  lambda s: not (s.get("ac") == "ON" and s.get("humidifier") == "ON" and s.get("humidity_high_flag") is True),
                  "ac and humidifier not both ON during high humidity"),
    ]


def initial_state() -> State:
    return {
        "living_room_light": "OFF", "bedroom_light": "OFF", "hallway_light": "OFF",
        "thermostat": 70, "fan": "OFF", "heater": "OFF", "door_lock": "LOCKED",
        "window_blind": "CLOSED", "ac": "OFF", "humidifier": "OFF",
        "away_mode_flag": False, "humidity_high_flag": False,
    }


def env_auto_transitions():
    def to_away_mode(s):
        s2 = dict(s); s2["away_mode_flag"] = True; return s2
    def to_home(s):
        s2 = dict(s); s2["away_mode_flag"] = False; return s2
    def to_humid_high(s):
        s2 = dict(s); s2["humidity_high_flag"] = True; return s2
    def to_humid_normal(s):
        s2 = dict(s); s2["humidity_high_flag"] = False; return s2
    return [to_away_mode, to_home, to_humid_high, to_humid_normal]


class EventStream:
    def __init__(self, events: List[str], seed: int = 0, drift_rate: float = 0.0):
        self.events = events
        self.rng = np.random.default_rng(seed)
        self.n = len(events)
        self.base_probs = np.ones(self.n) / self.n
        self.drift_rate = drift_rate

    def _current_probs(self, t: int) -> np.ndarray:
        if self.drift_rate <= 0:
            return self.base_probs
        phase = np.linspace(0, 2 * np.pi, self.n, endpoint=False)
        drift = 1.0 + self.drift_rate * np.sin(self.drift_rate * t / 50.0 + phase)
        drift = np.clip(drift, 0.05, None)
        return drift / drift.sum()

    def sample(self, t: int) -> str:
        probs = self._current_probs(t)
        idx = self.rng.choice(self.n, p=probs)
        return self.events[idx]
