import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.stylegan import StyleGAN
from models.detector import DeepfakeDetector, TripletLossDetector, triplet_loss
from utils.data_utils import (
    get_dataloader,
    save_checkpoint,
    load_checkpoint,
    generate_fake_images,
    prepare_triplet_batch
)

def train_stylegan(generator, dataloader, num_epochs, device, save_dir):
    optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    criterion = nn.BCEWithLogitsLoss()
    
    writer = SummaryWriter(os.path.join(save_dir, 'logs'))
    
    for epoch in range(num_epochs):
        generator.train()
        total_loss = 0
        
        for batch in tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            real_images = batch.to(device)
            batch_size = real_images.size(0)
            
            # Gerçek görüntüler için etiketler
            real_labels = torch.ones(batch_size, 1).to(device)
            
            # Sahte görüntüler oluştur
            z = torch.randn(batch_size, 512).to(device)
            fake_images = generator(z)
            fake_labels = torch.zeros(batch_size, 1).to(device)
            
            # Generator loss
            g_loss = criterion(fake_images, real_labels)
            
            optimizer.zero_grad()
            g_loss.backward()
            optimizer.step()
            
            total_loss += g_loss.item()
        
        avg_loss = total_loss / len(dataloader)
        writer.add_scalar('Loss/Generator', avg_loss, epoch)
        
        if (epoch + 1) % 5 == 0:
            save_checkpoint(generator, optimizer, epoch, 
                          os.path.join(save_dir, f'generator_epoch_{epoch+1}.pt'))
            
            # Örnek görüntüler oluştur
            with torch.no_grad():
                sample_z = torch.randn(16, 512).to(device)
                sample_images = generator(sample_z)
                writer.add_images('Generated Images', sample_images, epoch)
    
    writer.close()

def train_detector(detector, generator, dataloader, num_epochs, device, save_dir):
    optimizer = optim.Adam(detector.parameters(), lr=0.0001)
    criterion = nn.BCELoss()
    
    writer = SummaryWriter(os.path.join(save_dir, 'logs'))
    
    for epoch in range(num_epochs):
        detector.train()
        total_loss = 0
        
        for batch in tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            real_images = batch.to(device)
            batch_size = real_images.size(0)
            
            # Sahte görüntüler oluştur
            fake_images = generate_fake_images(generator, batch_size, device)
            
            # Triplet loss için batch hazırla
            anchor, positive, negative = prepare_triplet_batch(real_images, fake_images, batch_size)
            
            # Özellik çıkar
            _, anchor_emb = detector(anchor)
            _, positive_emb = detector(positive)
            _, negative_emb = detector(negative)
            
            # Triplet loss hesapla
            trip_loss = triplet_loss(anchor_emb, positive_emb, negative_emb)
            
            # Sınıflandırma loss
            real_preds, _ = detector(real_images)
            fake_preds, _ = detector(fake_images)
            
            real_labels = torch.ones_like(real_preds)
            fake_labels = torch.zeros_like(fake_preds)
            
            cls_loss = criterion(real_preds, real_labels) + criterion(fake_preds, fake_labels)
            
            # Toplam loss
            loss = trip_loss + cls_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        writer.add_scalar('Loss/Detector', avg_loss, epoch)
        
        if (epoch + 1) % 5 == 0:
            save_checkpoint(detector, optimizer, epoch,
                          os.path.join(save_dir, f'detector_epoch_{epoch+1}.pt'))
    
    writer.close()

def main():
    # Konfigürasyon
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = 'data/ffhq'
    save_dir = 'checkpoints'
    num_epochs = 100
    batch_size = 32
    
    # Model oluştur
    generator = StyleGAN().to(device)
    detector = TripletLossDetector().to(device)
    
    # DataLoader oluştur
    dataloader = get_dataloader(data_dir, batch_size=batch_size)
    
    # Eğitim
    print("StyleGAN eğitimi başlıyor...")
    train_stylegan(generator, dataloader, num_epochs, device, save_dir)
    
    print("Detector eğitimi başlıyor...")
    train_detector(detector, generator, dataloader, num_epochs, device, save_dir)

if __name__ == '__main__':
    main() 