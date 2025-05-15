import torch
import torch.nn as nn
import torchvision.models as models

class DeepfakeDetector(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        # ResNet50'yi temel model olarak kullan
        self.resnet = models.resnet50(pretrained=pretrained)
        
        # Son fully connected katmanını değiştir
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.resnet(x)

class TripletLossDetector(nn.Module):
    def __init__(self, embedding_dim=512):
        super().__init__()
        # ResNet50'yi özellik çıkarıcı olarak kullan
        self.resnet = models.resnet50(pretrained=True)
        num_features = self.resnet.fc.in_features
        
        # Özellik çıkarıcı
        self.feature_extractor = nn.Sequential(
            nn.Linear(num_features, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # Sınıflandırıcı
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        features = self.resnet.conv1(x)
        features = self.resnet.bn1(features)
        features = self.resnet.relu(features)
        features = self.resnet.maxpool(features)
        
        features = self.resnet.layer1(features)
        features = self.resnet.layer2(features)
        features = self.resnet.layer3(features)
        features = self.resnet.layer4(features)
        
        features = self.resnet.avgpool(features)
        features = torch.flatten(features, 1)
        
        embeddings = self.feature_extractor(features)
        predictions = self.classifier(embeddings)
        
        return predictions, embeddings

def triplet_loss(anchor, positive, negative, margin=1.0):
    """
    Triplet loss hesaplama
    anchor: Referans örnek
    positive: Pozitif örnek (aynı sınıf)
    negative: Negatif örnek (farklı sınıf)
    margin: Triplet loss için margin değeri
    """
    distance_positive = torch.norm(anchor - positive, p=2, dim=1)
    distance_negative = torch.norm(anchor - negative, p=2, dim=1)
    losses = torch.relu(distance_positive - distance_negative + margin)
    return losses.mean() 