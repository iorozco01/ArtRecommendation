import os
import torch
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import json
# Load the pre-trained Inception-v3 model
model = models.inception_v3(pretrained=True)
model.eval()  # Set to evaluation mode

# Define image preprocessing
preprocess = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_features(img_path):
    img = Image.open(img_path).convert('RGB')
    img_tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(img_tensor)
    return features.squeeze().cpu().numpy()

def load_metadata(metadata_path):
    with open(metadata_path, 'r') as f:
        return json.load(f)

def process_artwork(image_id, image_folder, metadata_folder):
    img_path = os.path.join(image_folder, f"{image_id}.jpg")
    metadata_path = os.path.join(metadata_folder, f"{image_id}.json")
    
    features = extract_features(img_path)
    metadata = load_metadata(metadata_path)
    
    return {
        'id': image_id,
        'features': features,
        'metadata': metadata
    }

def process_all_artworks(image_folder, metadata_folder):
    artworks = []
    for filename in os.listdir(image_folder):
        if filename.endswith('.jpg'):
            image_id = filename.split('.')[0]
            artwork = process_artwork(image_id, image_folder, metadata_folder)
            artworks.append(artwork)
    return artworks

image_folder = 'met_images'
metadata_folder = 'met_metadata'
all_artworks = process_all_artworks(image_folder, metadata_folder)