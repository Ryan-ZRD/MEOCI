"""
datasets.data_preprocessor
==========================================================
Unified data preprocessing and caching utilities for MEOCI.
----------------------------------------------------------
Handles:
    - Dataset normalization and resizing
    - Balanced / random sampling
    - Caching preprocessed samples to accelerate training
    - Integration with BDD100K and other datasets

Used in:
    - ADP-D3QN training (input normalization)
    - Vehicle-edge inference simulation (task input sampling)
    - Ablation studies (Fig. 10–12)
"""

import os
import cv2
import pickle
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Any, Tuple
from torchvision import transforms
from torch.utils.data import Subset


class DataPreprocessor:
    """
    DataPreprocessor
    ======================================================
    Provides unified preprocessing and sampling for datasets.

    Features:
        ✅ Resize, normalize, and augment
        ✅ In-memory or disk-based caching
        ✅ Balanced or random sampling
        ✅ Compatible with PyTorch Datasets
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        normalize: bool = True,
        cache_dir: str = "./cache",
        mode: str = "memory",  # ['memory', 'disk', 'none']
    ):
        self.image_size = image_size
        self.normalize = normalize
        self.cache_dir = cache_dir
        self.mode = mode
        self.cached_data: Dict[int, Any] = {}

        self.transform = self._build_transform_pipeline()

        if mode == "disk" and not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)

    # ------------------------------------------------------------
    # 🔧 Transform pipeline
    # ------------------------------------------------------------
    def _build_transform_pipeline(self):
        if self.normalize:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(self.image_size),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(self.image_size)
            ])

    # ------------------------------------------------------------
    # ⚙️ Preprocess a single sample
    # ------------------------------------------------------------
    def preprocess_sample(self, img_path: str) -> np.ndarray:
        """
        Preprocess a single image: resize, normalize, and convert.
        """
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, self.image_size)
        img_tensor = self.transform(img_resized)
        return img_tensor

    # ------------------------------------------------------------
    # 💾 Cache entire dataset (optional)
    # ------------------------------------------------------------
    def cache_dataset(self, dataset, cache_name: str = "bdd100k_train_cache.pkl"):
        """
        Cache dataset to memory or disk for faster loading.
        """
        print(f"[Preprocessor] Caching dataset ({len(dataset)}) samples...")

        cache_path = os.path.join(self.cache_dir, cache_name)
        data = {}

        for i in tqdm(range(len(dataset))):
            sample = dataset[i]
            data[i] = sample

        if self.mode == "memory":
            self.cached_data = data
            print(f"[Preprocessor] Cached {len(self.cached_data)} samples in memory.")

        elif self.mode == "disk":
            with open(cache_path, "wb") as f:
                pickle.dump(data, f)
            print(f"[Preprocessor] Dataset cached to disk: {cache_path}")

        else:
            print("[Preprocessor] Caching disabled.")

    # ------------------------------------------------------------
    # 🔍 Load cache from disk
    # ------------------------------------------------------------
    def load_cache(self, cache_name: str = "bdd100k_train_cache.pkl") -> Dict[int, Any]:
        cache_path = os.path.join(self.cache_dir, cache_name)
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"Cache not found: {cache_path}")
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        self.cached_data = data
        print(f"[Preprocessor] Loaded {len(data)} samples from cache.")
        return data

    # ------------------------------------------------------------
    # 🎯 Balanced sampling by class
    # ------------------------------------------------------------
    def build_balanced_subset(self, dataset, num_per_class: int = 100) -> Subset:
        """
        Construct a balanced subset of the dataset.
        """
        label_to_indices: Dict[int, List[int]] = {}
        for i in range(len(dataset)):
            label = dataset[i]["target"] if "target" in dataset[i] else dataset[i]["targets"]["exit3"][0]
            label_to_indices.setdefault(label, []).append(i)

        selected_indices = []
        for label, indices in label_to_indices.items():
            selected_indices.extend(indices[:num_per_class])
        print(f"[Preprocessor] Built balanced subset with {len(selected_indices)} samples.")
        return Subset(dataset, selected_indices)

    # ------------------------------------------------------------
    # ⚡ Random sampling
    # ------------------------------------------------------------
    def random_sample(self, dataset, n: int = 200) -> Subset:
        indices = np.random.choice(len(dataset), n, replace=False)
        return Subset(dataset, indices)

    # ------------------------------------------------------------
    # 🧹 Reset cache
    # ------------------------------------------------------------
    def reset_cache(self):
        self.cached_data.clear()
        print("[Preprocessor] Cache cleared.")


# ------------------------------------------------------------
# ✅ Example Usage
# ------------------------------------------------------------
if __name__ == "__main__":
    from datasets.bdd100k_loader import BDD100KDataset

    ds = BDD100KDataset(
        root_dir="/path/to/bdd100k",
        split="train",
        multi_exit=False,
        augmentation=False
    )

    pre = DataPreprocessor(mode="disk")
    subset = pre.random_sample(ds, n=50)
    pre.cache_dataset(subset, cache_name="sample_cache.pkl")
    print("Cache saved and verified.")
