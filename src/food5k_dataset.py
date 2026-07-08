# Food-5k: binary food vs non-food classification
# Handles the common Kaggle structure:
#   split/food/*.jpg
#   split/non_food/*.jpg

import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

CLASS_NAMES_5K = ['food', 'non_food']


def get_transforms_5k(split: str) -> transforms.Compose:
    if split == 'training':
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2,
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


class Food5kDataset(Dataset):
    """
    Loads Food-5k images.
    Expected structure:
        root/training/food/*.jpg
        root/training/non_food/*.jpg
        root/validation/food/*.jpg
        root/evaluation/food/*.jpg
    Labels: food=0, non_food=1
    For dual-head: coarse_label == fine_label (binary task)
    """

    def __init__(self, root_dir: str, split: str = 'training'):
        self.transform = get_transforms_5k(split)
        self.samples   = []  # (path, fine_label, coarse_label)

        split_dir = os.path.join(root_dir, split)
        if not os.path.isdir(split_dir):
            raise ValueError(f'Split folder not found: {split_dir}')

        for folder in sorted(os.listdir(split_dir)):
            folder_path = os.path.join(split_dir, folder)
            if not os.path.isdir(folder_path):
                continue
            fname = folder.lower().replace(' ', '_')
            # food=0, anything with "non" in name=1
            class_id = 1 if 'non' in fname else 0
            for img in os.listdir(folder_path):
                if img.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((
                        os.path.join(folder_path, img),
                        class_id,
                        class_id   # coarse == fine for binary
                    ))

        food_n    = sum(1 for _, c, _ in self.samples if c == 0)
        nonfood_n = sum(1 for _, c, _ in self.samples if c == 1)
        print(f'Food5k [{split}]: {len(self.samples)} images  '
              f'(food={food_n}, non_food={nonfood_n})')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, fine, coarse = self.samples[idx]
        try:
            img = Image.open(path).convert('RGB')
            return self.transform(img), fine, coarse
        except Exception:
            return self.__getitem__((idx + 1) % len(self.samples))