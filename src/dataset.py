import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from src.coarse_mapping import get_coarse_label

class Food11Dataset(Dataset):
    def __init__(self, data_dir, split='training', transform=None):
        self.data_dir = os.path.join(data_dir, split)
        self.transform = transform
        self.samples = []
        
        # Mapping class names to index (0-10)
        self.classes = sorted([d for d in os.listdir(self.data_dir) 
                              if os.path.isdir(os.path.join(self.data_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # Build list of (path, fine_label, coarse_label)
        for cls_name in self.classes:
            cls_path = os.path.join(self.data_dir, cls_name)
            for img_name in os.listdir(cls_path):
                if img_name.endswith('.jpg'):
                    path = os.path.join(cls_path, img_name)
                    fine_label = self.class_to_idx[cls_name]
                    coarse_label = get_coarse_label(cls_name)
                    self.samples.append((path, fine_label, coarse_label))
                    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, fine, coarse = self.samples[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, fine, coarse