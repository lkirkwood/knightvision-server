# ===== Knight Vision - Manual Testing Pipeline =====
import sys, os
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Extend path to access local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

# Core detection logic
from chess_detector import detect_chess_board

# Visual debugging tools
from visualisation_utils import (
    compare_images,
    map_bounding_boxes,
    map_corner_projection
)

# ===== Configuration =====
IMAGE_ID = "38_jpg.rf.5721f8bc9a1eb379ec89b72cece2017f"
model_path = "./knight_vision_detector.pt"
original_image_path = f"./examples/original_images/{IMAGE_ID}.jpg"
corrected_image_path = f"./examples/perspective_corrections/{IMAGE_ID}.jpg"
cumulative_homography_matrix_path = f"./examples/perspective_corrections/{IMAGE_ID}.npy"
orientation = "bottom"  # Options: "top", "bottom", "left", "right"

# ===== Load Model & Inputs =====
model = YOLO(model_path)
original_image = Image.open(original_image_path)
corrected_image = Image.open(corrected_image_path)
homography_matrix = np.load(cumulative_homography_matrix_path)

# ===== Run Detection =====
print(f"Model Classes: {model.names}")
print(">>> Running Detection and Generating FEN...")

result = detect_chess_board(
    model=model,
    original_image=original_image,
    corrected_image=corrected_image,
    homography_matrix=homography_matrix,
    orientation=orientation
)

# ===== Display Output =====
print(f"\n=== GENERATED FEN ===\n{result.get_fen()}")
print(f" \nCumulative Homography Matrix {homography_matrix}")

# ===== Visualisation =====
print("\n>>> Visualising Detection + Board Mapping")
compare_images(original_image, corrected_image)
map_bounding_boxes(original_image, result.raw_detections, "Original Image")
map_bounding_boxes(corrected_image, result.corrected_detections,  "Corrected Image")
map_corner_projection(original_image, corrected_image, homography_matrix)