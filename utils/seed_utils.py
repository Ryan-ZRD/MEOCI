

import os
import random
import numpy as np
import torch



def set_global_seed(seed: int = 42, deterministic: bool = True) -> None:

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    print(f"[SeedUtils] Global random seed set to {seed} (deterministic={deterministic})")



class temp_seed:
    """
    Temporarily override global random seed within context.

    Example:
        >>> with temp_seed(123):
        >>>     x = np.random.rand()
    """

    def __init__(self, seed: int):
        self.seed = seed
        self.state_random = None
        self.state_numpy = None
        self.state_torch = None

    def __enter__(self):
        # Save current RNG states
        self.state_random = random.getstate()
        self.state_numpy = np.random.get_state()
        self.state_torch = torch.get_rng_state()

        # Apply temporary seed
        set_global_seed(self.seed, deterministic=True)

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore RNG states
        random.setstate(self.state_random)
        np.random.set_state(self.state_numpy)
        torch.set_rng_state(self.state_torch)
        print(f"[SeedUtils] RNG state restored after temporary seed {self.seed}")


# ------------------------------------------------------------
# 🔄 Utility: Generate Seed List
# ------------------------------------------------------------
def generate_seed_list(base_seed: int, n: int) -> list:

    random.seed(base_seed)
    seeds = [random.randint(0, 99999) for _ in range(n)]
    print(f"[SeedUtils] Generated {n} seeds from base={base_seed}")
    return seeds



if __name__ == "__main__":
    set_global_seed(2025)

    # Normal random draws
    print("np:", np.random.rand(3))
    print("torch:", torch.rand(3))

    # Temporary seed override
    with temp_seed(123):
        print("Inside context:", np.random.rand(3))

    print("Restored RNG:", np.random.rand(3))
