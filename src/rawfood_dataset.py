# RawFoodDB: Raw Food Texture Database
# Source: https://mldta.com/dataset/rawfoot-db-raw-food-texture-database/
# Italian IVRL dataset — 68 categories, ~100 images each, lab lighting
# Dataset does NOT have predefined splits — we create 70/15/15 train/val/test

import os, random
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_transforms_raw(split: str) -> transforms.Compose:
    if split == 'train':
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            # Note: ColorJitter is important here because RawFoodDB has
            # 46 different lighting conditions — the model must be robust
            transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                   saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])


def build_rawfood_splits(root_dir: str, seed: int = 42,
                          train_ratio: float = 0.70,
                          val_ratio:   float = 0.15):
    """
    Scans root_dir for class subfolders, collects all images,
    and creates 70/15/15 train/val/test splits stratified per class.
    Returns: (train_samples, val_samples, test_samples, class_names)
    Each sample = (path, label_int)
    """
    random.seed(seed)
    class_folders = sorted([
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ])

    if len(class_folders) == 0:
        raise ValueError(f'No class folders found in: {root_dir}')

    class_to_idx = {name: i for i, name in enumerate(class_folders)}
    train_s, val_s, test_s = [], [], []

    for cls_name in class_folders:
        cls_path = os.path.join(root_dir, cls_name)
        imgs = [
            os.path.join(cls_path, f)
            for f in os.listdir(cls_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        ]
        random.shuffle(imgs)
        n      = len(imgs)
        n_tr   = max(1, int(n * train_ratio))
        n_val  = max(1, int(n * val_ratio))
        label  = class_to_idx[cls_name]
        train_s += [(p, label) for p in imgs[:n_tr]]
        val_s   += [(p, label) for p in imgs[n_tr:n_tr + n_val]]
        test_s  += [(p, label) for p in imgs[n_tr + n_val:]]

    print(f'RawFoodDB: {len(class_folders)} classes found')
    print(f'  Train: {len(train_s)}  Val: {len(val_s)}  Test: {len(test_s)}')
    return train_s, val_s, test_s, class_folders


class RawFoodDataset(Dataset):
    """
    Dataset wrapper for RawFoodDB splits.
    For dual-head: since all 68 categories are raw food textures, we
    create 5 coarse nutritional groups matching the dietary framework.
    """

    # Coarse mapping for RawFoodDB — 5 groups based on food type
    # You will refine this once you see the actual 68 class names
    COARSE_GROUPS = {
        'default': 0  # placeholder — updated in Kaggle after seeing class names
    }

    def __init__(self, samples: list, split: str, class_names: list):
        self.samples    = samples   # list of (path, fine_label)
        self.transform  = get_transforms_raw(split)
        self.class_names = class_names
        self.num_classes = len(class_names)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, fine_label = self.samples[idx]
        try:
            img = Image.open(path).convert('RGB')
            img = self.transform(img)
            # For RawFoodDB, coarse == fine (no dietary grouping available
            # without seeing actual class names — update after first run)
            return img, fine_label, fine_label
        except Exception:
            return self.__getitem__((idx + 1) % len(self.samples))