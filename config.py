# Configuration file for Image Captioning App

# Model configuration
MODEL_CONFIG = {
    'vocab_size': 8000,
    'embed_dim': 512,
    'num_heads': 16,
    'num_layers': 4,
    'ff_dim': 2048,
    'max_length': 50,
    'dropout': 0.1
}

# Google Drive model file ID
GDRIVE_MODEL_ID = "1Yyfk7tnx-vrYqdmVY9JluZn2PqaK-6_W"

# Paths
MODEL_PATH = "models/model.pth"
VOCAB_PATH = "vocabulary.pkl"

# Image processing settings
IMAGE_SIZE = (224, 224)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# App settings
MAX_UPLOAD_SIZE = 50  # MB
SUPPORTED_FORMATS = ['png', 'jpg', 'jpeg']

# Device settings
USE_GPU = True  # Set to False to force CPU usage
