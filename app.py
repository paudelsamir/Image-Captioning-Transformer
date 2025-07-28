import streamlit as st
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io
import time
import requests

# Import our custom modules
from model_loader import load_model_and_vocab_from_gdrive
from image_processing import load_feature_extractor, get_image_transforms, extract_image_features, generate_caption

# Page configuration
st.set_page_config(
    page_title="AI Image Captioning",
    page_icon="🆘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #1f77b4, #6a11cb, #2575fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        color: transparent;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #333;
        margin-bottom: 1rem;
    }
    .caption-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .info-box {
        background: #fff;
        color: #222;
        padding: 1.2rem 1rem 1rem 1rem;
        border-radius: 10px;
        border: 1px solid #1f77b4;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(31,119,180,0.07);
    }
    .author-signature {
        font-size: 1rem;
        color: #888;
        text-align: center;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Title and description
    st.markdown('<h1 class="main-header">AI Image Captioning with Transformers</h1>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        <p style="font-size:1.2rem; font-weight:600; color:#1f77b4; margin-bottom:0.5rem;">Welcome to the AI Image Captioning App!</p>
        <p style="color:#222; font-size:1.05rem;">This application uses a custom-trained Transformer model to generate descriptive captions for your images. Upload an image and watch as AI analyzes and describes what it sees!</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar with information
    with st.sidebar:
        st.markdown("""
        <div style="background:#f0f2f6; border-radius:8px; padding:1rem; margin-bottom:1rem; color:#222;">
            <b>Model:</b> Transformer + ResNet-18<br>
            <b>Vocabulary Size:</b> 7,234 words<br>
            <b>BLEU-4 Score:</b> 0.18
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("Model Parameters"):
            st.write("- **Embedding Dimension:** 512")
            st.write("- **Attention Heads:** 8")
            st.write("- **Decoder Layers:** 3")
            st.write("- **Feed Forward Dimension:** 2048")
            st.write("- **Dropout:** 0.2")
        
        with st.expander("Usage Tips"):
            st.write("- Upload clear, well-lit images")
            st.write("- Works best with common objects and scenes")
            st.write("- Images are automatically resized to 224x224")
            
        with st.expander("Model Limitations"):
            st.write("- Trained on specific dataset, may not recognize all objects")
            st.write("- Performance varies with image quality and lighting")
            st.write("- May generate generic descriptions for complex scenes")
            st.write("- Limited to vocabulary of 7,234 words")
        
        st.markdown('<div class="author-signature">Author: <a href="https://x.com/samireey" target="_blank" style="color:#e53935;">'
                '<span style="color:#e53935;">@samir</span></a></div>', unsafe_allow_html=True)

    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<h2 class="sub-header">Upload Your Image</h2>', unsafe_allow_html=True)
        
        # File uploader
        uploaded_file = st.file_uploader(
            "*choose an image file*", 
            type=['png', 'jpg', 'jpeg'],
            help="Supported formats: PNG, JPG, JPEG"
        )
        
        # Sample images section
        st.markdown("### Or try a sample image:")
        sample_images = {
            "Dog": "https://images.unsplash.com/photo-1587300003388-59208cc962cb?w=300",
            "Beach": "https://images.unsplash.com/photo-1507525428034-b723cf961d3e?w=300",
            "City": "https://images.unsplash.com/photo-1449824913935-59a10b8d2000?w=300"
        }
        
        selected_sample = st.selectbox("Select a sample image:", ["None"] + list(sample_images.keys()))
        
        if selected_sample != "None":
            st.image(sample_images[selected_sample], caption=f"Sample: {selected_sample}", width=300)
    
    with col2:
        st.markdown('<h2 class="sub-header"> Caption</h2>', unsafe_allow_html=True)
        st.markdown("*captions will appear here after image upload*")

        if uploaded_file is not None or selected_sample != "None":
            # Load the image
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", width=400)
            else:
                # Load the sample image from URL
                import requests
                try:
                    response = requests.get(sample_images[selected_sample])
                    image = Image.open(io.BytesIO(response.content))
                    st.image(image, caption=f"Sample Image: {selected_sample}", width=400)
                except Exception as e:
                    st.error(f"Error loading sample image: {e}")
                    image = None
            
            if image is not None:
                # Generate caption button
                if st.button("Generate Caption", type="primary"):
                    with st.spinner("Analyzing image..."):
                        try:
                            # Load models and vocabulary
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            status_text.text("Loading models...")
                            progress_bar.progress(25)
                            
                            # Load the trained model and vocabulary
                            decoder, word2idx, idx2word = load_model_and_vocab_from_gdrive()
                            progress_bar.progress(75)
                            
                            # Load feature extractor and transforms
                            feature_extractor = load_feature_extractor()
                            transform = get_image_transforms()
                            
                            status_text.text("Processing image...")
                            
                            # Extract image features
                            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                            image_features = extract_image_features(image, feature_extractor, transform, device)
                            
                            # Generate caption
                            status_text.text("Generating caption...")
                            caption = generate_caption(image_features, decoder, word2idx, idx2word, device=device)
                            
                            progress_bar.progress(100)
                            status_text.text("Caption generated successfully!")
                            
                            # Display the result in minimal gradient text
                            st.markdown(f"""
                            <div class="caption-box">
                                <h3 style="font-size:1.3rem; font-weight:600; background: linear-gradient(90deg, #1f77b4, #6a11cb, #2575fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; color: transparent; margin: 1rem 0 0.5rem 0;">{caption}</h3>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Clear progress indicators
                            time.sleep(1)
                            progress_bar.empty()
                            status_text.empty()
                            
                        except Exception as e:
                            st.error(f"An error occurred: {str(e)}")
                            st.info("Make sure all model files are properly loaded and the image is valid.")
        else:
            st.info("Please upload an image or select a sample to generate a caption.")
    
    # ...section removed for minimal UI...

if __name__ == "__main__":
    main()
