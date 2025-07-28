# Image Captioning 
A simple app that generates captions for images using a Transformer decoder and ResNet-18 features. Upload your own image or try a sample to see what the model describes!

---

## Project Notebook
- [Preprocessing, Training & Full Pipeline Notebook](https://github.com/paudelsamir/365DaysOfData/blob/main/10-Projects-Based-ML-DL/03-Image-Captioning/image-captioning.ipynb)

## Live Demo
- [Try it on Streamlit Cloud](https://image-captioning-samir.streamlit.app)
https://github.com/user-attachments/assets/de1d986e-57d5-4062-857f-04338a4ef4ba
---

## Model Info

- **Feature Extractor:** ResNet-18 (pretrained)
- **Decoder:** Transformer (3 layers, 8 heads, 512 emb, 2048 ff, dropout 0.2)
- **Vocabulary:** 7,234 words
- **Metric:** BLEU-4 score: 0.18

---

## Model Download

- [Model weights (Google Drive)](https://drive.google.com/file/d/1Yyfk7tnx-vrYqdmVY9JluZn2PqaK-6_W/view?usp=sharing)
- [Vocabulary (Google Drive)](https://drive.google.com/file/d/17QDWhwp6wQweaaHRk8FZmippeQ04m0gH/view?usp=drive_link)

*The app will auto-download these when you run it, so you don't need to do it manually unless you want to.*

---

## How to Run Locally

1. **Clone this repo:**
   ```bash
   git clone https://github.com/paudelsamir/Image-Captioning-Transformer.git
   cd Image-Captioning-Transformer
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the app:**
   ```bash
   streamlit run app.py
   streamlit run demo_app.py (no requirements needed)
   ```

### Windows Users
```bash
# Run the setup script
setup.bat
```

### Linux/Mac Users
```bash
# Make setup script executable and run
chmod +x setup.sh
./setup.sh
```
---

## Author
- [@samir](https://x.com/samireey)

---

*This is a fun project for learning and demo purposes. For details, see the notebook above.*

