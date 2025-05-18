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


def generate_full_fen(board, orientation):
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


# ===== FEN Generation =====
def detections_to_fen(detections, orientation):
    board = [["" for _ in range(8)] for _ in range(8)]
    yolo_to_fen = {
        "white-pawn": "P", "white-rook": "R", "white-knight": "N",
        "white-bishop": "B", "white-queen": "Q", "white-king": "K",
        "black-pawn": "p", "black-rook": "r", "black-knight": "n",
        "black-bishop": "b", "black-queen": "q", "black-king": "k"
    }
    for detected_piece in detections:
        row, col = detected_piece["row"], detected_piece["col"]
        piece_label = detected_piece["label"]
        piece_name = yolo_to_fen.get(piece_label.lower(), "")
        board[row][col] = piece_name
    canonical_board = rotate_board_to_fen_view(board)
    return canonical_board, generate_full_fen(canonical_board, orientation)


# ===== Visualisation (for Debugging) =====
def visualize_grid(img, detections, H, orientation):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img)
    grid_points = []
    for i in range(9):
        row = []
        for j in range(9):
            pt = np.array([j, i, 1]).reshape(3, 1)
            mapped_pt = np.dot(np.linalg.inv(H), pt)
            mapped_pt /= mapped_pt[2]
            row.append((mapped_pt[0][0], mapped_pt[1][0]))
        grid_points.append(row)

    for i in range(9):
        ax.plot(
            [p[0] for p in grid_points[i]],
            [p[1] for p in grid_points[i]],
            color="white",
            linewidth=2,
        )
    for j in range(9):
        ax.plot(
            [grid_points[i][j][0] for i in range(9)],
            [grid_points[i][j][1] for i in range(9)],
            color="white",
            linewidth=2,
        )

    for j in range(8):
        mid_x = (grid_points[8][j][0] + grid_points[7][j][0]) / 2
        mid_y = (grid_points[8][j][1] + grid_points[7][j][1]) / 2
        square = grid_to_square(0, j, orientation)
        ax.text(
            mid_x + 20,
            mid_y + 30,
            square[0],
            ha="center",
            va="top",
            color="black",
            fontsize=12,
            fontweight="bold",
        )

    for i in range(8):
        mid_x = (grid_points[i][0][0] + grid_points[i][1][0]) / 2
        mid_y = (grid_points[i][0][1] + grid_points[i][1][1]) / 2
        square = grid_to_square(i, 0, orientation)
        ax.text(
            mid_x - 25,
            mid_y + 20,
            square[1],
            ha="right",
            va="center",
            color="black",
            fontsize=12,
            fontweight="bold",
        )

    for det in detections:
        cx, cy = det["center"]
        row, col = det["grid"]
        board_coord = grid_to_square(col, row, orientation)
        ax.plot(cx, cy, "o", color="red", markersize=10)
        ax.text(
            cx + 5,
            cy,
            board_coord,
            color="yellow",
            fontsize=10,
            fontweight="bold",
            bbox=dict(facecolor="black", alpha=0.6),
        )

    plt.title("Knight Vision - Detected Pieces on Board Grid")
    plt.axis("off")
    plt.show()


# ===== Main API-Friendly Wrapper =====
class ChessDetectionResult:
    def __init__(self, image, detections, homography, orientation):
        self.image = image
        self.detections = detections
        self.homography = homography
        self.orientation = orientation
        canonical_board, self.fen = detections_to_fen(detections, orientation)
        self.fen = generate_full_fen(canonical_board, orientation)

    def visualize(self):
        visualize_grid(self.image, self.detections, self.homography, self.orientation)


def detect_chess_board(model: YOLO, img: Image.Image, orientation):
    processed_img = img.convert("RGB").resize(TARGET_SIZE, Image.Resampling.LANCZOS)
    # H = homography matrix from Board Isolation
    results = model.predict(img, imgsz=640)

    detections = []
    for result in results:
        if result.boxes is None:
            raise RuntimeError("Model found no boxes in the image!")

        for box, conf, cls in zip(
            result.boxes.xyxy, result.boxes.conf, result.boxes.cls
        ):
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2
            cy = y2 - 0.25 * (y2 - y1)

            grid_x, grid_y = map_point_homography((cx.item(), cy.item()), H)

            # Skip detections outside the playable 8x8 grid
            if not (0 <= grid_x < 8 and 0 <= grid_y < 8):
                print(f"[SKIPPED] Detection at ({cx.item():.1f}, {cy.item():.1f}) mapped to ({grid_x:.2f}, {grid_y:.2f}) → outside board")
                continue

            col = min(max(int(grid_x), 0), 7)
            row = min(max(int(grid_y), 0), 7)

            row, col = transform_position(row, col, orientation)

            detections.append({
                "row": row,
                "col": col,
                "grid": (row, col),
                "center": (cx.item(), cy.item()),
                "label": result.names[int(cls)],
                "confidence": conf.item(),
            })

    return ChessDetectionResult(processed_img, detections, H, orientation)


# Example Usage:
# result = detect_chess_board("board.jpg", "left")
# print(result.fen)
# result.visualize()
