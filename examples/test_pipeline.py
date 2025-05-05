# ===== Knight Vision - Testing Pipeline (Board Info Selector + Chess Detector) =====

from board_info_selector import select_corners, select_orientation
from chess_detector import detect_chess_board
import sys

# ===== Input image path =====
image_path = "./yolo_data/test/images/IMG_0159_JPG.rf.f0d34122f8817d538e396b04f2b70d33.jpg"

# ===== Step 1: Select orientation (user clicks) =====
# For now manually set this, later can automate or make GUI dropdown
orientation = "left"   # Possible values: "bottom", "top", "left", "right"

# ===== Step 2: Run detection and get FEN =====
print(">>> Running detection and generating FEN...")
result = detect_chess_board(image_path, orientation)

# ===== Step 3: Output FEN =====
print("\n=== GENERATED FEN ===")
print(result.fen)

# ===== (Optional) Step 4:  Visualise result =====
print("\n>>> Visualising detection + board mapping...")
result.visualize()