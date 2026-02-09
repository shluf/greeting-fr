#!/bin/bash

# Face Recognition Greeting System - Installation Script
# Copyright (c) 2026

set -e

echo "=================================="
echo "Face Recognition Greeting Installer"
echo "=================================="
echo ""

# Check if Python3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed. Please install Python3 first."
    exit 1
fi

echo "✓ Python3 found"

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Installing pip..."
    sudo apt-get update
    sudo apt-get install -y python3-pip
fi

echo "✓ pip3 found"

# Install system dependencies
echo ""
echo "📦 Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y cmake build-essential libopenblas-dev liblapack-dev libx11-dev libgtk-3-dev alsa-utils ffmpeg

# Install Python packages
echo ""
echo "📦 Installing Python packages..."
pip3 install --upgrade pip
pip3 install -r requirements.txt

# Create directory structure
echo ""
echo "📁 Creating directory structure..."
mkdir -p images
mkdir -p models/face
mkdir -p models/tts
echo "✓ Directory structure created"

# Check if Piper model exists
if [ ! -f "models/tts/id_ID-news_tts-medium.onnx" ]; then
    echo ""
    echo "⚠️  Piper TTS model not found!"
    echo "Please download the model from:"
    echo "https://huggingface.co/rhasspy/piper-voices/tree/main/id/id_ID/news_tts/medium"
    echo ""
    echo "Or run these commands:"
    echo "wget -P models/tts/ https://huggingface.co/rhasspy/piper-voices/resolve/main/id/id_ID/news_tts/medium/id_ID-news_tts-medium.onnx"
    echo "wget -P models/tts/ https://huggingface.co/rhasspy/piper-voices/resolve/main/id/id_ID/news_tts/medium/id_ID-news_tts-medium.onnx.json"
else
    echo "✓ Piper TTS model found"
fi

# Check if dataset exists
if [ ! -f "models/face/dataset_faces.dat" ]; then
    echo ""
    echo "⚠️  Face dataset not found!"
    echo "Please add face images to the 'images/' folder and run:"
    echo "python3 add_face.py"
fi

echo ""
echo "=================================="
echo "✅ Installation completed!"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. Add face images to 'images/' folder (e.g., Budi_Santoso.jpg)"
echo "2. Run: python3 add_face.py"
echo "3. Run: python3 main.py"
echo ""
echo "Press 'q' to quit the application."
echo ""
