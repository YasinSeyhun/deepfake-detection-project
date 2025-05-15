import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path

class FFHQDataset(Dataset):
    def __init__(
        self,
        root: str,
        resolution: int = 1024,
        transform=None,
    ):
        self.root = Path(root)
        self.resolution = resolution
        self.transform = transform or transforms.Compose([
            transforms.Resize(resolution),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        
        # Get all image files
        self.image_files = sorted([
            f for f in self.root.glob("**/*.png")
            if f.is_file()
        ])
        
        if not self.image_files:
            raise RuntimeError(f"No images found in {root}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a random image if loading fails
            image = torch.randn(3, self.resolution, self.resolution)
            return image
        
        if self.transform:
            image = self.transform(image)
        
        return image 