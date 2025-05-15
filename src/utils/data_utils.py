import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

class FFHQDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image

def get_dataloader(data_dir, batch_size=32, num_workers=4, shuffle=True):
    dataset = FFHQDataset(data_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader

def save_checkpoint(model, optimizer, epoch, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, filename)

def load_checkpoint(model, optimizer, filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch']

def generate_fake_images(generator, num_images, device):
    generator.eval()
    with torch.no_grad():
        z = torch.randn(num_images, 512).to(device)
        fake_images = generator(z)
    return fake_images

def prepare_triplet_batch(real_images, fake_images, batch_size):
    """
    Triplet loss için batch hazırlama
    """
    indices = torch.randperm(len(real_images))[:batch_size]
    anchor = real_images[indices]
    
    # Pozitif örnekler (gerçek görüntüler)
    pos_indices = torch.randperm(len(real_images))[:batch_size]
    positive = real_images[pos_indices]
    
    # Negatif örnekler (sahte görüntüler)
    neg_indices = torch.randperm(len(fake_images))[:batch_size]
    negative = fake_images[neg_indices]
    
    return anchor, positive, negative 