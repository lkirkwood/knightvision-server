# ===== Knight Vision - Manual Testing Pipeline =====


from chess_detector import detect_chess_board, ChessDetectionResult
from ultralytics import YOLO
from PIL import Image
import numpy as np

# (Optional Debugging Utilities)
from visualisation_utils import (
    show_image_comparison,
    draw_detections_on_image,
    visualise_on_digital_board
)


# ===== Setup =====
model_path = "./model/best.pt"
original_image_path = "./examples/test_board_original.png"
corrected_image_path = "./examples/test_board_corrected.jpg"
homography_matrix_path = "./examples/homography_matrix.npy"
orientation = "left"   # or "bottom", "top", "right"

# ===== Load Inputs =====
model = YOLO(model_path)
original_image = Image.open(original_image_path)
corrected_image = Image.open(corrected_image_path)
homography_matrix = np.load(homography_matrix_path)

# ===== Check Model is Valid (Optional) =====
print(f"Model Classes: {model.names}")

# ===== Run Detection + FEN Generation =====
print(">>> Running Detection and Generating FEN...")

chess_detection_result = detect_chess_board(
    model=model,
    original_image=original_image,
    corrected_image=corrected_image,
    homography_matrix=homography_matrix,
    orientation=orientation
)

# ===== Display Generated FEN =====
print("\n=== GENERATED FEN ===")
print(chess_detection_result.fen)

# ===== Visualisation(s) =====
print("\n>>> Visualising Detection + Board Mapping")
show_image_comparison(original_image, corrected_image)
draw_detections_on_image(original_image, chess_detection_result.detections, title="Detections on Original Image")
draw_detections_on_image(corrected_image, chess_detection_result.detections, title="Detections on Corrected Image")
visualise_on_digital_board(chess_detection_result.detections)