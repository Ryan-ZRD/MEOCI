"""
datasets.bdd100k_loader
==========================================================
BDD100K dataset loader and preprocessor for MEOCI framework.
----------------------------------------------------------
This module handles:
    - Image and annotation loading
    - Preprocessing (resize, normalization, augmentation)
    - Dataset splitting (train/val/test)
    - Multi-exit inference label generation

References:
    - BDD100K Dataset: https://bdd-data.berkeley.edu/
    - Paper Section 5.2: "Experimental Settings"
"""

import os
import cv2
import json
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Dict, Tuple, List, Any


class BDD100KDataset(Dataset):
    """
    Custom dataset for BDD100K images and labels.
    Supports both classification and detection-based multi-exit inference.
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        img_size: Tuple[int, int] = (224, 224),
        multi_exit: bool = True,
        augmentation: bool = True,
        cache: bool = False,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.multi_exit = multi_exit
        self.augmentation = augmentation
        self.cache = cache

        # Paths
        self.image_dir = os.path.join(root_dir, "images", split)
        self.label_path = os.path.join(root_dir, "labels", f"{split}.json")

        # Load annotations
        with open(self.label_path, "r") as f:
            self.annotations = json.load(f)

        # Prepare image list
        self.img_files = list(self.annotations.keys())

        # Transformation pipeline
        self.transform = self._build_transform_pipeline()

        # Optional cache
        self.cached_data = {}
        if self.cache:
            self._cache_dataset()

    # ------------------------------------------------------------
    # 🔧 Build transformation pipeline
    # ------------------------------------------------------------
    def _build_transform_pipeline(self):
        if self.augmentation:
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(self.img_size),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
        else:
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
        return transform

    # ------------------------------------------------------------
    # 🧠 Label processing for classification/detection
    # ------------------------------------------------------------
    def _parse_label(self, ann: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract target labels (object category, confidence, etc.)
        """
        if "labels" not in ann:
            return {"class_id": 0, "confidence": 1.0}

        objects = ann["labels"]
        if len(objects) == 0:
            return {"class_id": 0, "confidence": 1.0}

        # Randomly select one object (for lightweight classification mode)
        obj = random.choice(objects)
        category = obj.get("category", "car")
        confidence = obj.get("score", 1.0)
        class_id = self._map_label_to_id(category)

        return {"class_id": class_id, "confidence": confidence}

    @staticmethod
    def _map_label_to_id(category: str) -> int:
        """
        Map BDD100K categories to integer IDs
        """
        mapping = {
            "car": 0, "bus": 1, "truck": 2, "person": 3,
            "bicycle": 4, "motorcycle": 5, "traffic light": 6, "traffic sign": 7
        }
        return mapping.get(category.lower(), 8)  # others → 8

    # ------------------------------------------------------------
    # 🧩 Multi-exit supervision logic
    # ------------------------------------------------------------
    def _generate_exit_targets(self, label: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate pseudo supervision signals for multi-exit classifiers.
        """
        base_class = label["class_id"]
        conf = label["confidence"]

        # Simulate varying exit confidence thresholds
        exits = {
            "exit1": (base_class, conf * 0.7),
            "exit2": (base_class, conf * 0.85),
            "exit3": (base_class, conf),
        }
        return exits

    # ------------------------------------------------------------
    # 🖼️ Get sample
    # ------------------------------------------------------------
    def __getitem__(self, idx: int):
        if self.cache and idx in self.cached_data:
            return self.cached_data[idx]

        img_name = self.img_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        ann = self.annotations[img_name]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = self.transform(img)

        label = self._parse_label(ann)

        if self.multi_exit:
            exits = self._generate_exit_targets(label)
            sample = {
                "image": img_tensor,
                "targets": exits,
                "filename": img_name,
            }
        else:
            sample = {
                "image": img_tensor,
                "target": label["class_id"],
                "filename": img_name,
            }

        if self.cache:
            self.cached_data[idx] = sample
        return sample

    # ------------------------------------------------------------
    # 🔢 Length
    # ------------------------------------------------------------
    def __len__(self):
        return len(self.img_files)

    # ------------------------------------------------------------
    # ⚡ Dataset caching (optional)
    # ------------------------------------------------------------
    def _cache_dataset(self):
        print(f"[BDD100K] Caching {len(self.img_files)} samples to memory...")
        for i in range(len(self.img_files)):
            self.__getitem__(i)
        print("[BDD100K] Cache complete.")


# ------------------------------------------------------------
# ✅ Utility: build dataloader
# ------------------------------------------------------------
def build_bdd100k_dataloader(
    root_dir: str,
    split: str = "train",
    batch_size: int = 16,
    num_workers: int = 4,
    shuffle: bool = True,
    **kwargs
) -> DataLoader:
    """
    Build DataLoader for BDD100K dataset.
    """
    dataset = BDD100KDataset(root_dir=root_dir, split=split, **kwargs)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    print(f"[BDD100K] Loaded {split} set ({len(dataset)} samples)")
    return loader


# ------------------------------------------------------------
# ✅ Example test
# ------------------------------------------------------------
if __name__ == "__main__":
    data_root = "/path/to/bdd100k"
    loader = build_bdd100k_dataloader(
        root_dir=data_root,
        split="train",
        batch_size=4,
        multi_exit=True,
        augmentation=True
    )
    for i, batch in enumerate(loader):
        print(f"Batch {i}: {batch['image'].shape}")
        print("Exits:", batch["targets"])
        if i == 1:
            break
