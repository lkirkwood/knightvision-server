:: Knight Vision Inference Environment Setup (CPU-Only)
:: ======================================================
:: This script installs only what is needed to run chess_detector_lw.py
:: Ensure you're inside a virtual environment before running this.

@echo off

:: Step 1 - Install YOLOv8 framework
pip install ultralytics==8.3.113

:: Step 2 - Install PyTorch + TorchVision (CPU-only builds)
pip install torch==2.2.0+cpu torchvision==0.17.0+cpu --index-url https://download.pytorch.org/whl/cpu

:: Step 3 - Core utilities
pip install opencv-python pillow numpy==1.26.4 pyyaml

echo ===============================================
echo Knight Vision lightweight environment installed (CPU)
echo ===============================================
