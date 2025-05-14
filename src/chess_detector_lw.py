from ultralytics import YOLO
import numpy as np
import cv2
from PIL import Image

# Constants
TARGET_SIZE = (416, 416)
TRAPEZIUM_CORNERS = np.array(
    [[74, 20], [316, 27], [371, 341], [37, 346]], dtype=np.float32
)


def compute_homography(source_points):
    target_points = np.array([[0, 0], [8, 0], [8, 8], [0, 8]], dtype=np.float32)
    H, _ = cv2.findHomography(source_points, target_points)
    return H


def map_point_homography(point, H):
    pt = np.array([point[0], point[1], 1]).reshape(3, 1)
    mapped_pt = np.dot(H, pt)
    mapped_pt /= mapped_pt[2]
    return mapped_pt[0][0], mapped_pt[1][0]


def rotate_grid(row, col, orientation):
    match orientation:
        case "bottom": return row, col
        case "top": return 7 - row, 7 - col
        case "left": return col, row
        case "right": return 7 - col, 7 - row
        case _: raise ValueError(f"Invalid orientation: {orientation}")


def to_fen_board_coords(board):
    return np.flipud(np.array(board)).tolist()


def generate_full_fen(board, orientation):
    fen_rows = []
    for row in board:
        fen_row = ""
        empty_count = 0
        for cell in row:
            if cell == "":
                empty_count += 1
            else:
                if empty_count > 0:
                    fen_row += str(empty_count)
                    empty_count = 0
                fen_row += cell
        if empty_count > 0:
            fen_row += str(empty_count)
        fen_rows.append(fen_row)
    return "/".join(fen_rows) + " w - - 0 1"
    # Uncomment the following line to allow castling by default
    # return "/".join(fen_rows) + " w KQkq - 0 1"


def detections_to_fen(detections, orientation):
    board = [["" for _ in range(8)] for _ in range(8)]
    yolo_to_fen = {
        "white-pawn": "P", "white-rook": "R", "white-knight": "N",
        "white-bishop": "B", "white-queen": "Q", "white-king": "K",
        "black-pawn": "p", "black-rook": "r", "black-knight": "n",
        "black-bishop": "b", "black-queen": "q", "black-king": "k"
    }
    for det in detections:
        row, col = det["row"], det["col"]
        label = det["label"]
        piece = yolo_to_fen.get(label.lower(), "")
        board[row][col] = piece
    canonical_board = to_fen_board_coords(board)
    return canonical_board, generate_full_fen(canonical_board, orientation)


class ChessDetectionResult:
    def __init__(self, image, detections, homography, orientation):
        self.image = image
        self.detections = detections
        self.homography = homography
        self.orientation = orientation
        canonical_board, self.fen = detections_to_fen(detections, orientation)


def detect_chess_board(model: YOLO, img: Image.Image, orientation):
    processed_img = img.convert("RGB").resize(TARGET_SIZE, Image.Resampling.LANCZOS)
    H = compute_homography(TRAPEZIUM_CORNERS)
    results = model.predict(processed_img, imgsz=640)

    detections = []
    for result in results:
        if result.boxes is None:
            raise RuntimeError("Model found no boxes in the image!")

        for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2
            cy = y2 - 0.25 * (y2 - y1)

            # Project into board coordinates
            grid_x, grid_y = map_point_homography((cx.item(), cy.item()), H)

            # Skip detections outside the playable 8x8 grid
            if not (0 <= grid_x < 8 and 0 <= grid_y < 8):
                print(f"[SKIPPED] Detection at ({cx.item():.1f}, {cy.item():.1f}) mapped to ({grid_x:.2f}, {grid_y:.2f}) → outside board")
                continue

            col = min(max(int(grid_x), 0), 7)
            row = min(max(int(grid_y), 0), 7)

            row, col = rotate_grid(row, col, orientation)

            detections.append({
                "row": row,
                "col": col,
                "grid": (row, col),
                "center": (cx.item(), cy.item()),
                "label": result.names[int(cls)],
                "confidence": conf.item()
            })

    return ChessDetectionResult(processed_img, detections, H, orientation)
