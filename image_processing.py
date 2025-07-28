import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import streamlit as st

@st.cache_resource
def load_feature_extractor():
    """Load and return the ResNet feature extractor"""
    resnet = models.resnet18(pretrained=True)
    feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
    feature_extractor.eval()
    return feature_extractor

def get_image_transforms():
    """Get image preprocessing transforms"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def extract_image_features(image, feature_extractor, transform, device='cpu'):
    """Extract features from an image using ResNet"""
    # Convert PIL image if needed
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Apply transforms
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # Extract features
    feature_extractor = feature_extractor.to(device)
    with torch.no_grad():
        features = feature_extractor(img_tensor)
        features = features.view(features.size(0), -1)  # Flatten
    
    return features

def generate_caption(image_features, decoder, word2idx, idx2word, max_length=50, device='cpu'):
    """Generate a caption for given image features"""
    decoder.eval()
    decoder = decoder.to(device)
    
    # Start with <start> token
    caption = ['<start>']
    
    for _ in range(max_length):
        # Convert current caption to token indices
        tokens = [word2idx.get(word, word2idx['<unk>']) for word in caption]
        tokens_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
        
        # Generate the next word
        with torch.no_grad():
            outputs = decoder(tokens_tensor, image_features)
            next_token = outputs[0, -1].argmax(dim=-1).item()
        
        # Convert token index to word
        next_word = idx2word[next_token]
        caption.append(next_word)
        
        # Stop if end token is generated
        if next_word == '<end>':
            break
    
    # Remove start and end tokens and join words
    generated_caption = ' '.join(caption[1:-1])
    
    # Clean up the caption
    generated_caption = generated_caption.replace('<end>', '').strip()
    
    return generated_caption
