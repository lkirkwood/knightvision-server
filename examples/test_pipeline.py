# ===== Knight Vision - Testing Pipeline (Board Info Selector + Chess Detector) =====

from board_info_selector import select_corners, select_orientation
from chess_detector import detect_chess_board
import sys

# Debugging
from ultralytics import YOLO
from PIL import Image

# ===== Input image path =====
model_path = "./model/best.pt"
image_path = "./examples/test_board.png"

model = YOLO(model_path)
img = Image.open(image_path)

# ===== Step 1: Select orientation (user clicks) =====
# For now manually set this, later can automate or make GUI dropdown
orientation = "left"   # Possible values: "bottom", "top", "left", "right"

# ===== Step 2: Run detection and get FEN =====
print(f"Model Classes: {model.names}")
print(">>> Running detection and generating FEN...")
result = detect_chess_board(model=model, img=img, orientation=orientation)

# ===== Step 3: Output FEN =====
print("\n=== GENERATED FEN ===")
print(result.fen)

# ===== (Optional) Step 4:  Visualise result =====
print("\n>>> Visualising detection + board mapping...")
result.visualize()