#!/bin/bash

echo "================================"
echo "Image Captioning App Setup"
echo "================================"
echo

echo "Installing Python dependencies..."
pip install -r requirements.txt

echo
echo "Setup complete!"
echo
echo "To run the demo app (no model required):"
echo "streamlit run demo_app.py"
echo
echo "To run the full app (requires trained model):"
echo "streamlit run app.py"
echo
echo "================================"
