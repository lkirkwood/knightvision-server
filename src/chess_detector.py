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
        self.orientation=orientation
        
    # def remap_square_index(self, row, col):
    #     """Adjusts (row, col) based on board orientation."""
    #     match self.orientation:
    #         case "bottom": return row, col
    #         case "top": return row, col
    #         case "left": return row, col
    #         case "right": return row, col

    def get_square_id(self, row, col):
        """
        Returns algebraic notation (e.g., 'e4') for a square at (row, col),
        adjusted for the current board orientation.
        """
        if self.orientation == "bottom":
            file = "abcdefgh"[col]
            rank = "12345678"[7 - row]
        elif self.orientation == "top":
            file = "hgfedcba"[col]
            rank = "87654321"[7 - row]
        elif self.orientation == "left":
            file = "12345678"[row]
            rank = "abcdefgh"[col]
        elif self.orientation == "right":
            file = "87654321"[row]
            rank = "hgfedcba"[col]
        else:
            raise ValueError(f"Invalid orientation: {self.orientation}")

        return f"{file}{rank}"
        
    def rotate_board_to_fen_view(self, board):
        """Rotates or flips the entire board to match FEN 'white-on-bottom' orientation."""
        arr = np.array(board)
        match self.orientation:
            case "bottom": return arr.tolist()                  # Don't Move   (Already Correct)
            case "top": return np.flipud(arr).tolist()          # Rotate 180°  (Flip Vertically)
            case "left": return np.rot90(arr, k=1).tolist()     # Rotate 90°   (Counter-Clockwise)
            case "right": return np.rot90(arr, k=3).tolist()    # Rotate 90°   (Clockwise)



def detections_to_board(detections):
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
    homogenous_point  = np.array([point[0], point[1], 1]).reshape(3, 1)

    # Apply homography matrix to warp the point into corrected space
    warped_point = np.dot(homography_matrix, homogenous_point)

    # Normalise by third (scale) coordinate to convert back to 2D Cartesian Coordinates
    warped_point /= warped_point[2]

    # Extract the individual corrected coordinates
    corrected_x = warped_point[0][0]
    corrected_y = warped_point[1][0]

    return corrected_x, corrected_y

def debug_log_bounding_box_mappings(
    detections: list,
    homography_matrix: np.ndarray
):
    """
    Logs the full bounding box mapping pipeline for every detection,
    from raw bbox to final grid assignment (row, col).
    Assumes orientation adjustment is handled later (e.g., before FEN generation).
    """
    print("=== Bounding Box Mapping Debug ===")
    for i, det in enumerate(detections):
        label = det["label"]
        conf = det["confidence"]
        x1, y1, x2, y2 = det["box"]

        # Step 1: Center points
        center_x = (x1 + x2) / 2
        center_y = y2 - 0.25 * (y2 - y1)
        print(f"\n[{i+1}] {label} ({conf:.2f})")
        print(f"Raw box:      ({x1:.2f}, {y1:.2f}) → ({x2:.2f}, {y2:.2f})")
        print(f"Raw center:   ({(x1 + x2)/2:.2f}, {(y1 + y2)/2:.2f})")
        print(f"Offset center (lower): ({center_x:.2f}, {center_y:.2f})")

        # Step 2: Project via homography
        corrected_x, corrected_y = map_point_to_board_space((center_x, center_y), homography_matrix)
        print(f"Projected:    ({corrected_x:.2f}, {corrected_y:.2f})")

        # Step 3: Map to 8x8 grid
        grid_col = min(max(int(corrected_x // 64), 0), 7)
        grid_row = min(max(int(corrected_y // 64), 0), 7)
        print(f"Mapped to grid square: row={grid_row}, col={grid_col}")




# ===== Main API-Friendly Wrapper =====
class ChessDetectionResult:
    def __init__(self, image, raw_detections, corrected_detections, homography_matrix, orientation):
        self.image = image
        self.raw_detections = raw_detections  # Detections before homography
        self.corrected_detections = corrected_detections  # For projection-based logic
        self.homography_matrix = homography_matrix
        self.orientation = orientation

        # Convert corrected detections to board grid
        self.raw_board = detections_to_board(corrected_detections)

        print("[DEBUG] Board Before Rotation:")
        for row in self.raw_board:
            print(" ".join(cell if cell else "." for cell in row))

        # Apply rotation (if any) to align with white-on-bottom FEN format
        self.rotated_board = BoardOrientation(orientation).rotate_board_to_fen_view(self.raw_board)
        self.fen = board_to_fen(self.rotated_board)

    def get_board(self, rotated=True):
        """Returns the board grid — rotated for FEN view, or raw layout"""
        return self.rotated_board if rotated else self.raw_board

    def get_fen(self):
        """Returns FEN string"""
        return self.fen

        


def detect_chess_board(
        model: YOLO,
        original_image: Image.Image,
        corrected_image: Image.Image = None,
        homography_matrix: np.ndarray = None,
        orientation: str = "left",
        confidence_threshold: float = 0.5
):
    #print(f"[DEBUG] Image Sizes - Original: {original_image.size}, Corrected: {corrected_image.size if corrected_image else 'N/A'}")

    # === Ensure image used for YOLO is same as one used to generate homography ===
    #if original_image.size != (512, 512):
       # print("[WARN] Resizing original image to 512x512 for consistency")
        #original_image = original_image.resize((512, 512), Image.Resampling.LANCZOS)

    predictions = model.predict(original_image.convert("RGB"), imgsz=512)

    corrected_detections = []  # renamed for clarity
    raw_detections = []        # optional: store untransformed bounding boxes (in original image space)

    board_orientation = BoardOrientation(orientation)

    for prediction in predictions:
        if prediction.boxes is None:
            raise RuntimeError("Model Found No Boxes in the Image!")
        


        for bbox, confidence, class_id in zip(
            prediction.boxes.xyxy, prediction.boxes.conf, prediction.boxes.cls
        ):
            if float(confidence) < confidence_threshold:
                print(f"[SKIPPED] {model.names[int(class_id)]} ({confidence:.2f}) below threshold {confidence_threshold}")
                continue
            
            label = model.names[int(class_id)]

            min_x, min_y, max_x, max_y = map(float, bbox)
            center_x = (min_x + max_x) / 2
            center_y = max_y - 0.25 * (max_y - min_y)

            # Map center + box via homography
            corrected_cx, corrected_cy = map_point_to_board_space((center_x, center_y), homography_matrix)
            # if not (0 <= corrected_cx < 512 and 0 <= corrected_cy < 512):
            #     print(f"[SKIPPED] {model.names[int(class_id)]} (Out of Bounds: X = {corrected_cx}, Y = {corrected_cy}) below threshold {confidence_threshold}")
            #     continue
            
            #grid_col = min(max(int(corrected_cx // 64), 0), 7)
            #grid_row = min(max(int(corrected_cy // 64), 0), 7)
            grid_col = int(corrected_cx // 64)
            grid_row = int(corrected_cy // 64)

            # Transform full box corners
            corners = [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]
            warped = [map_point_to_board_space(pt, homography_matrix) for pt in corners]
            xs, ys = zip(*warped)
            corrected_box = [min(xs), min(ys), max(xs), max(ys)]


            raw_detections.append({
                "row": None,   # Unknown in distorted space
                "col": None,   # Unknown in distorted space
                "grid": None,  # Unknown in distorted space
                "center": (center_x, center_y),
                "label": label,
                "confidence": float(confidence),
                "box": [min_x, min_y, max_x, max_y],
            })

            corrected_detections.append({
                "row": grid_row,
                "col": grid_col,
                "grid": (grid_row, grid_col),
                "center": (corrected_cx, corrected_cy),
                "label": label,
                "confidence": float(confidence),
                "box": corrected_box,
            })

    debug_log_bounding_box_mappings(corrected_detections, homography_matrix)

    return ChessDetectionResult(
        image=original_image,
        raw_detections=raw_detections,
        corrected_detections=corrected_detections,
        homography_matrix=homography_matrix,
        orientation=orientation
    )
