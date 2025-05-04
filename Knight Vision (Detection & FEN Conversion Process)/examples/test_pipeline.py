# ===== Knight Vision - Testing Pipeline (Board Info Selector + Chess Detector) =====

from board_info_selector import select_corners, select_orientation
from chess_detector import detect_chess_board
import sys

# ===== Input image path =====
image_path = "./yolo_data/test/images/IMG_0159_JPG.rf.f0d34122f8817d538e396b04f2b70d33.jpg"

# ===== Step 1: Select corners (user clicks) =====
print(">>> Please select the 4 board corners...")
corners = select_corners(image_path)

# ===== Step 2: Select orientation (user clicks) =====
print(">>> Please select orientation of White's side...")
orientation = select_orientation(image_path)

# ===== Step 3: Run detection and get FEN =====
print(">>> Running detection and generating FEN...")
result = detect_chess_board(image_path, corners, orientation)

# ===== Step 4: Output FEN =====
print("\n=== GENERATED FEN ===")
print(result.fen)



# ===== (Optional) Step 5:  Visualise result =====
print("\n>>> Visualising detection + board mapping...")
result.visualize()
