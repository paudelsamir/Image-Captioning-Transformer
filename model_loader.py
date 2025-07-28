import os
import torch
import streamlit as st
import gdown
import pickle
from model import TransformerDecoder  # make sure this matches your actual file

@st.cache_resource
def load_model_and_vocab_from_gdrive():
    """Load both model and vocabulary from Google Drive"""
    MODEL_FILE_ID = "1Yyfk7tnx-vrYqdmVY9JluZn2PqaK-6_W"  # Replace with your model file ID
    VOCAB_FILE_ID = "17QDWhwp6wQweaaHRk8FZmippeQ04m0gH"  # Your vocabulary file ID
    
    MODEL_PATH = "models/best_decoder.pth"
    VOCAB_PATH = "models/vocabulary.pkl"
    
    # make directories
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    # Download model if needed
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading model from Google Drive...")
        model_url = f"https://drive.google.com/uc?id={MODEL_FILE_ID}"
        gdown.download(model_url, MODEL_PATH, quiet=False)
        st.success("Model downloaded!")

    # Download vocabulary if needed
    if not os.path.exists(VOCAB_PATH):
        st.info("Downloading vocabulary from Google Drive...")
        vocab_url = f"https://drive.google.com/uc?id={VOCAB_FILE_ID}"
        gdown.download(vocab_url, VOCAB_PATH, quiet=False)
        st.success("Vocabulary downloaded!")

    # Load vocabulary
    with open(VOCAB_PATH, 'rb') as f:
        vocab_data = pickle.load(f)
    
    vocab = vocab_data['vocab']
    word2idx = vocab_data['word2idx']
    idx2word = vocab_data['idx2word']
    vocab_size = len(vocab)

    # Initialize model with correct parameters from training
    model = TransformerDecoder(
        vocab_size=vocab_size,
        embed_dim=512,
        num_heads=8,
        num_layers=3,  # Updated to match your training
        ff_dim=2048,
        max_length=50,
        dropout=0.2
    )

    # Load the checkpoint and extract the model state dict
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(checkpoint['model_state_dict'])  # Access the nested state dict
    model.eval()
    
    return model, word2idx, idx2word

@st.cache_resource
def load_model_from_gdrive():
    """Legacy function for backward compatibility"""
    model, _, _ = load_model_and_vocab_from_gdrive()
    return model
