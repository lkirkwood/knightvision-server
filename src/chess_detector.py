# chess_detector.py

from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from itertools import groupby

# Constants
TARGET_SIZE = (512, 512)

# ===== Custom Board Orientation Handler =====
class BoardOrientation:
    def __init__(self, orientation: str):
        valid_orientations = {"top", "bottom", "left", "right"}
        if orientation not in valid_orientations:
            raise ValueError(f"Invalid Orientation: {orientation}")
        
    def remap_square_index(self, row, col):
        """Adjusts (row, col) based on board orientation."""
        match self.orientation:
            case "bottom": return row, col
            case "top": return 7 - row, 7 - col
            case "left": return col, row
            case "right": return 7 - col, 7 - row

    def get_square_id(self, row, col):
        """Returns algebraic notation (e.g., 'e4') for a square at (row, col) based on orientation."""
        row, col = self.remap_square_index(row, col)
        files = ["a", "b", "c", "d", "e", "f", "g", "h"]
        ranks = ["1", "2", "3", "4", "5", "6", "7", "8"]
        return f"{ranks[col]}{files[row]}"
        
    def rotate_board_to_fen_view(self, board):
        """Rotates or flips the entire board to match FEN 'white-on-bottom' orientation."""
        arr = np.array(board)
        match self.orientation:
            case "bottom": return arr.tolist()                  # Don't Move   (Already Correct)
            case "top": return np.flipud(arr).tolist()          # Rotate 180°  (Flip Vertically)
            case "left": return np.rot90(arr, k=1).tolist()     # Rotate 90°   (Counter-Clockwise)
            case "right": return np.rot90(arr, k=3).tolist()    # Rotate 90°   (Clockwise)



def detections_to_board(detections, orientation):
    board = [["" for _ in range(8)] for _ in range(8)]
    yolo_to_fen_mapping = {
        "white-pawn": "P", "white-rook": "R", "white-knight": "N",
        "white-bishop": "B", "white-queen": "Q", "white-king": "K",
        "black-pawn": "p", "black-rook": "r", "black-knight": "n",
        "black-bishop": "b", "black-queen": "q", "black-king": "k"
    }
    for detected_piece in detections:
        row, col = detected_piece["row"], detected_piece["col"]
        piece_label = detected_piece["label"]
        piece_name = yolo_to_fen_mapping.get(piece_label.lower(), "")
        board[row][col] = piece_name
    return board


def board_to_fen(board: list[list[str]]) -> str:
    fen_rows = []
    for row in board:
        fen_row = ""
        empty_cells_in_row = 0
        for cell in row:
            if cell == "":
                empty_cells_in_row += 1
            else:
                if empty_cells_in_row > 0:
                    fen_row += str(empty_cells_in_row)
                    empty_cells_in_row = 0
                fen_row += cell
        if empty_cells_in_row > 0:
            fen_row += str(empty_cells_in_row)
        fen_rows.append(fen_row)
    piece_placement = "/".join(fen_rows)
    castling_options = "-"
    fen = f"{piece_placement} w {castling_options} - 0 1"
    return fen


def map_point_to_board_space(point: tuple[float, float], homography_matrix: np.ndarray) -> tuple[float, float]:
    """Projects a point (x, y) from the original image space into the 512x512 perspective-corrected board space."""
    # Convert 2D pixel-point to Homogeneous Coordinates for matrix projection ((x,y) → (x, y, 1))
    homogenous_point  = np.array([point[0], point[1]], 1).reshape(3, 1)
    
    # Apply homography matrix to warp the point into corrected space
    warped_point = np.dot(homography_matrix, homogenous_point)

    # Normalise by third (scale) coordinate to convert back to 2D Cartesian Coordinates
    warped_point /= warped_point[2]

    # Extract the individual corrected coordinates
    corrected_x = warped_point[0][0]
    corrected_y = warped_point[1][0]

    return corrected_x, corrected_y


# ===== Main API-Friendly Wrapper =====
class ChessDetectionResult:
    def __init__(self, image, detections, homography_matrix):
        self.image = image
        self.detections = detections
        self.homography_matrix = homography_matrix
        board = detections_to_board(detections)
        self.fen = board_to_fen(board)
        self.board = board


def detect_chess_board(
        model: YOLO,
        original_image: Image.Image,
        corrected_image: Image.Image = None,
        homography_matrix: np.ndarray = None,
        orientation: str = "left"
):
    # YOLO predicts on the original image (with distortion). It resizes internally to 416×416, 
    # but returns box coordinates in original pixel space, matching the homography's source domain.

    predictions = model.predict(original_image.convert("RGB"), imgsz=416)

    board_orientation = BoardOrientation(orientation)
    detections = []
    
    for prediction in predictions:
        if prediction.boxes is None:
            raise RuntimeError("Model Found No Boxes in the Image!")

        for bounding_box, confidence, class_id in zip(prediction.boxes.xyxy, prediction.boxes.conf, prediction.boxes.cls):
            min_x, min_y, max_x, max_y = bounding_box
            center_x = (min_x + max_x) / 2
            center_y = max_y - 0.25 * (max_y - min_y) # Lower center point (more reliable)

            # Project the center of the bounding box into the perspective-corrected board space
            corrected_x, corrected_y = map_point_to_board_space((center_x.item(), center_y.item()), homography_matrix)

            # Skip points that fall outside the playable 8×8 board region
            if not (0 <= corrected_x < 512 and 0 <= corrected_y < 512):
                print(
                    f"[SKIPPED] Detection at ({center_x:.1f}, {center_y:.1f}) "
                    f"→ ({corrected_x:.1f}, {corrected_y:.1f}) is outside board bounds"
                )
                continue
            
            # Map to 8×8 grid coordinates (each square = 64×64 pixels)
            grid_col = int(corrected_x // 64)
            grid_row = int(corrected_y // 64)
            row, col = board_orientation.remap_square_index(grid_row, grid_col)

            detections.append({
                "row": row,
                "col": col,
                "grid": (row, col),
                "center": (center_x.item(), center_y.item()),
                "label": prediction.names[int(class_id)],
                "confidence": confidence.item(),
            })

    return ChessDetectionResult(original_image, detections, homography_matrix)