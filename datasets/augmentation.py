"""
datasets.augmentation
==========================================================
Custom data augmentation utilities for MEOCI framework.
----------------------------------------------------------
Supports two augmentation profiles:
    1. LightAugment   — for real-time edge inference
    2. StrongAugment  — for robust offline model training

Used in:
    - BDD100KDataset (datasets/bdd100k_loader.py)
    - Multi-exit model training and evaluation
    - Edge offloading simulations (noise / lighting variation)
"""

import random
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image, ImageEnhance


# ------------------------------------------------------------
# 🎛️ Random Gaussian Blur
# ------------------------------------------------------------
class RandomGaussianBlur:
    """Randomly apply Gaussian blur with variable kernel size."""

    def __init__(self, p: float = 0.3):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            ksize = random.choice([3, 5])
            img = cv2.GaussianBlur(np.array(img), (ksize, ksize), 0)
            return Image.fromarray(img)
        return img


# ------------------------------------------------------------
# 🌧️ Random Weather Effects (fog, rain, motion blur)
# ------------------------------------------------------------
class RandomWeatherEffect:
    """Simulate environmental distortions for vehicular data."""

    def __init__(self, p: float = 0.2):
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img

        effect_type = random.choice(["fog", "rain", "motion_blur"])
        img_np = np.array(img, dtype=np.uint8)

        if effect_type == "fog":
            overlay = np.full_like(img_np, 255, dtype=np.uint8)
            alpha = random.uniform(0.1, 0.3)
            img_np = cv2.addWeighted(img_np, 1 - alpha, overlay, alpha, 0)

        elif effect_type == "rain":
            for _ in range(random.randint(100, 200)):
                x1 = random.randint(0, img_np.shape[1] - 1)
                y1 = random.randint(0, img_np.shape[0] - 1)
                length = random.randint(5, 15)
                thickness = random.randint(1, 2)
                color = (200, 200, 200)
                cv2.line(img_np, (x1, y1), (x1 + 2, y1 + length), color, thickness)

        elif effect_type == "motion_blur":
            kernel_size = random.choice([5, 7, 9])
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
            kernel /= kernel_size
            img_np = cv2.filter2D(img_np, -1, kernel)

        return Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8))


# ------------------------------------------------------------
# 🔆 LightAugment: lightweight augmentations for inference
# ------------------------------------------------------------
def LightAugment(image_size=(224, 224)):
    """
    Lightweight augmentation suitable for low-latency edge inference.
    """
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomHorizontalFlip(p=0.3),
        RandomGaussianBlur(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


# ------------------------------------------------------------
# ⚡ StrongAugment: robust augmentations for training
# ------------------------------------------------------------
def StrongAugment(image_size=(224, 224)):
    """
    Heavier augmentation strategy for robust model generalization.
    """
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=10)
        ], p=0.7),
        RandomWeatherEffect(p=0.3),
        RandomGaussianBlur(p=0.4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


# ------------------------------------------------------------
# 🧪 Example usage
# ------------------------------------------------------------
if __name__ == "__main__":
    from PIL import Image

    img = Image.open("/path/to/sample.jpg").convert("RGB")

    light_aug = LightAugment()
    strong_aug = StrongAugment()

    img_light = light_aug(img)
    img_strong = strong_aug(img)

    print("Light augment tensor shape:", img_light.shape)
    print("Strong augment tensor shape:", img_strong.shape)
