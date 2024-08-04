import torch
from torchvision import models, transforms
from PIL import Image

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

# Function to extract features
def extract_features(img_path):
    img = Image.open(img_path)
    img_tensor = preprocess(img)
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        features = model(img_tensor)
    return features

