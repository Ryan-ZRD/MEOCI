"""
datasets.split_dataset
==========================================================
Dataset splitting utility for MEOCI framework.
----------------------------------------------------------
Splits datasets (e.g., BDD100K) into train/val/test subsets
with reproducible ratios and optional stratified sampling.

Used in:
    - Pretraining & finetuning of MEOCI multi-exit models
    - Ablation studies (Fig. 8–10)
    - Environment simulation input division
"""

import os
import json
import random
from typing import List, Dict, Tuple
from sklearn.model_selection import train_test_split


def split_dataset(
    dataset_root: str,
    output_dir: str = None,
    ratios: Tuple[float, float, float] = (0.7, 0.2, 0.1),
    stratify: bool = True,
    seed: int = 42,
    save_format: str = "json"
) -> Dict[str, List[str]]:
    """
    Split dataset into train / val / test sets.

    Args:
        dataset_root (str): Root directory containing images and labels.
        output_dir (str): Where to save split files.
        ratios (tuple): (train, val, test) ratio.
        stratify (bool): Whether to use label-based stratification.
        seed (int): Random seed for reproducibility.
        save_format (str): 'json' or 'txt'

    Returns:
        dict: {
            "train": [...],
            "val": [...],
            "test": [...]
        }
    """
    assert sum(ratios) <= 1.0, "Sum of split ratios must be <= 1.0"

    random.seed(seed)

    img_dir = os.path.join(dataset_root, "images")
    label_path = os.path.join(dataset_root, "labels", "train.json")
    with open(label_path, "r") as f:
        labels = json.load(f)

    img_files = list(labels.keys())
    classes = []

    for name in img_files:
        anns = labels[name].get("labels", [])
        if len(anns) == 0:
            classes.append("unknown")
        else:
            cls = anns[0].get("category", "car")
            classes.append(cls)

    # Stratified or random split
    if stratify:
        train_files, val_test_files, y_train, y_val_test = train_test_split(
            img_files, classes, test_size=(1 - ratios[0]), stratify=classes, random_state=seed
        )
        val_ratio = ratios[1] / (ratios[1] + ratios[2])
        val_files, test_files, _, _ = train_test_split(
            val_test_files, y_val_test, test_size=(1 - val_ratio), stratify=y_val_test, random_state=seed
        )
    else:
        train_files, val_test_files = train_test_split(
            img_files, test_size=(1 - ratios[0]), random_state=seed
        )
        val_ratio = ratios[1] / (ratios[1] + ratios[2])
        val_files, test_files = train_test_split(val_test_files, test_size=(1 - val_ratio), random_state=seed)

    splits = {"train": train_files, "val": val_files, "test": test_files}

    # Save
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        if save_format == "json":
            out_path = os.path.join(output_dir, "splits.json")
            with open(out_path, "w") as f:
                json.dump(splits, f, indent=2)
        else:
            for name, files in splits.items():
                out_path = os.path.join(output_dir, f"{name}.txt")
                with open(out_path, "w") as f:
                    f.writelines([fn + "\n" for fn in files])

        print(f"[SplitDataset] Saved splits to: {output_dir}")

    print(f"[SplitDataset] Done! Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")
    return splits


# ------------------------------------------------------------
# ✅ Utility: load split files
# ------------------------------------------------------------
def load_splits(path: str) -> Dict[str, List[str]]:
    """
    Load existing dataset splits.
    """
    if path.endswith(".json"):
        with open(path, "r") as f:
            return json.load(f)
    elif os.path.isdir(path):
        splits = {}
        for name in ["train", "val", "test"]:
            file_path = os.path.join(path, f"{name}.txt")
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    splits[name] = [line.strip() for line in f.readlines()]
        return splits
    else:
        raise ValueError(f"Invalid split file path: {path}")


# ------------------------------------------------------------
# 🧪 Example Usage
# ------------------------------------------------------------
if __name__ == "__main__":
    dataset_root = "/path/to/bdd100k"
    output_dir = "./splits"

    splits = split_dataset(
        dataset_root=dataset_root,
        output_dir=output_dir,
        ratios=(0.7, 0.2, 0.1),
        stratify=True,
        seed=42,
        save_format="json"
    )

    print("Sample:", {k: len(v) for k, v in splits.items()})
