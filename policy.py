"""far_iot.policy -- numpy-only LinUCB contextual bandit (reconstructed)."""
from __future__ import annotations
import numpy as np
from typing import List, Dict, Tuple
from .rules import Rule, State


def build_context(rho_i: Rule, rho_j: Rule, sigma: State, owners: List[str]) -> np.ndarray:
    owner_onehot = np.zeros(len(owners))
    if rho_i.owner in owners:
        owner_onehot[owners.index(rho_i.owner)] = 1.0
    is_numeric = 1.0 if rho_i.conflict_class == "numeric" else 0.0
    feat = np.concatenate([
        [rho_i.priority / 10.0, rho_j.priority / 10.0, (rho_i.priority - rho_j.priority) / 10.0],
        owner_onehot, [is_numeric, 1.0]
    ])
    return feat


class LinUCBPolicy:
    def __init__(self, dim: int, alpha: float = 1.0, seed: int = 0):
        self.dim = dim; self.alpha = alpha
        self.rng = np.random.default_rng(seed)
        self.A: Dict[str, np.ndarray] = {}
        self.b: Dict[str, np.ndarray] = {}

    def _ensure_arm(self, arm_id: str):
        if arm_id not in self.A:
            self.A[arm_id] = np.eye(self.dim)
            self.b[arm_id] = np.zeros(self.dim)

    def score(self, arm_id: str, x: np.ndarray) -> float:
        self._ensure_arm(arm_id)
        A_inv = np.linalg.inv(self.A[arm_id])
        theta_hat = A_inv @ self.b[arm_id]
        mean = float(theta_hat @ x)
        bonus = self.alpha * float(np.sqrt(x @ A_inv @ x))
        return mean + bonus

    def select(self, candidates: List[Tuple[str, np.ndarray]]) -> str:
        scores = [(arm_id, self.score(arm_id, x)) for arm_id, x in candidates]
        scores.sort(key=lambda t: t[1], reverse=True)
        return scores[0][0]

    def update(self, arm_id: str, x: np.ndarray, reward: float):
        self._ensure_arm(arm_id)
        self.A[arm_id] += np.outer(x, x)
        self.b[arm_id] += reward * x
