import base64
import os
import random
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from io import BytesIO


# Image preprocessing function
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# Base64 decoding function
def decode_base64_image(data):
    image_data = base64.b64decode(data)
    return Image.open(BytesIO(image_data)).convert('RGB')

# Model definition (ResNet-18 based outfit recommender)
class SimpleOutfitRecommender(nn.Module):
    def __init__(self):
        super(SimpleOutfitRecommender, self).__init__()
        
        # Feature extractor (ResNet-18)
        self.feature_extractor = models.resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-1])
        
        # Binary classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2, 256),  # 512 features for each item
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, shirt, pants):
        shirt_features = self.feature_extractor(shirt).view(-1, 512)
        pants_features = self.feature_extractor(pants).view(-1, 512)
        combined_features = torch.cat((shirt_features, pants_features), dim=1)
        return self.classifier(combined_features)
