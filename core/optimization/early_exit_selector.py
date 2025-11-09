import numpy as np
from typing import List, Dict, Optional


class EarlyExitSelector:
    """
    EarlyExitSelector
    ==========================================================
    Implements the multi-exit decision mechanism for collaborative
    inference models with early-exit branches.

    The selector dynamically determines which exit point to use
    for each inference task, balancing latency and accuracy.

    References:
        Section 3.4 - Early-Exit Point Selection
        Eq. (17)–(18) in the MEOCI paper
    """

    def __init__(self,
                 acc_threshold: float = 0.80,        # minimum acceptable accuracy
                 lambda_delay: float = 0.6,          # weight for latency
                 lambda_energy: float = 0.3,         # weight for energy
                 lambda_conf: float = 0.1,           # weight for confidence
                 energy_per_flop: float = 1e-9,
                 seed: int = 42):
        self.acc_threshold = acc_threshold
        self.lambda_delay = lambda_delay
        self.lambda_energy = lambda_energy
        self.lambda_conf = lambda_conf
        self.energy_per_flop = energy_per_flop
        self.random_state = np.random.RandomState(seed)

    # ------------------------------------------------------------
    # Compute exit score for one candidate
    # ------------------------------------------------------------
    def _compute_exit_score(self, exit_info: Dict) -> float:
        """
        Compute cost for each exit:
        J_i = λ1*T_i + λ2*E_i − λ3*Conf_i
        """
        T_i = exit_info["latency"]
        E_i = exit_info["energy"]
        C_i = exit_info["confidence"]
        cost = self.lambda_delay * T_i + self.lambda_energy * E_i - self.lambda_conf * C_i
        return cost

    # ------------------------------------------------------------
    # Select the best exit point
    # ------------------------------------------------------------
    def select_exit_point(self, exits: List[Dict]) -> Dict:
        """
        Select the optimal exit point based on latency–energy–confidence tradeoff.

        Args:
            exits: list of exits, each dict contains:
                   {"id", "latency", "accuracy", "confidence", "flops"}

        Returns:
            chosen exit dict
        """
        valid_exits = [e for e in exits if e["accuracy"] >= self.acc_threshold]
        if not valid_exits:
            # fallback to final exit
            valid_exits = [exits[-1]]

        # Compute energy for each exit (normalized)
        for e in valid_exits:
            e["energy"] = e["flops"] * self.energy_per_flop
            e["score"] = self._compute_exit_score(e)

        # Select minimal cost
        best = min(valid_exits, key=lambda x: x["score"])
        return best

    # ------------------------------------------------------------
    # Optional: adaptive accuracy threshold update
    # ------------------------------------------------------------
    def adapt_threshold(self, observed_acc: float):
        """
        Dynamically adjust accuracy threshold based on observed accuracy.
        """
        if observed_acc < self.acc_threshold:
            self.acc_threshold = max(0.7, self.acc_threshold - 0.01)
        else:
            self.acc_threshold = min(0.9, self.acc_threshold + 0.005)

    # ------------------------------------------------------------
    # Simulate multi-exit profile (for test)
    # ------------------------------------------------------------
    def simulate_exits(self, num_exits: int = 5) -> List[Dict]:
        exits = []
        for i in range(num_exits):
            latency = 10 + 8 * i + self.random_state.uniform(-2, 2)
            flops = 1e8 * (i + 1)
            acc = 0.6 + 0.08 * i + self.random_state.uniform(-0.02, 0.02)
            conf = 0.5 + 0.1 * i + self.random_state.uniform(-0.05, 0.05)
            exits.append({
                "id": i + 1,
                "latency": latency / 1000,  # convert ms → s
                "accuracy": acc,
                "confidence": conf,
                "flops": flops
            })
        return exits

    # ------------------------------------------------------------
    # Summary utility
    # ------------------------------------------------------------
    def summary(self, chosen_exit: Dict) -> Dict:
        return {
            "exit_id": chosen_exit["id"],
            "latency_ms": round(chosen_exit["latency"] * 1e3, 2),
            "accuracy": round(chosen_exit["accuracy"], 3),
            "confidence": round(chosen_exit["confidence"], 3),
            "energy_mJ": round(chosen_exit["energy"] * 1e3, 3),
            "score": round(chosen_exit["score"], 5),
        }


# ✅ Example quick test
if __name__ == "__main__":
    selector = EarlyExitSelector(acc_threshold=0.8)
    simulated = selector.simulate_exits(num_exits=5)
    chosen = selector.select_exit_point(simulated)
    print("Chosen exit:", selector.summary(chosen))
