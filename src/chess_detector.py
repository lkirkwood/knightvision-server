# chess_detector.py

from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from itertools import groupby

# Constants
TARGET_SIZE = (416, 416)
TRAPEZIUM_CORNERS = np.array(
    [
        [74, 20],  # Top Left
        [316, 27],  # Top Right
        [371, 341],  # Bottom Right
        [37, 346],  # Bottom Left
    ],
    dtype=np.float32,
)


# ===== Homography and Coordinate Mapping =====
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
        case "bottom":
            return row, col
        case "top":
            return 7 - row, 7 - col
        case "left":
            return col, row
        case "right":
            return 7 - col, 7 - row
        case _:
            raise ValueError(f"Invalid grid orientation: {orientation}")


def grid_to_square(row, col, orientation):
    """
    Converts grid coordinates (row, col) to chess square notation based on board orientation.
    """

    if orientation == "bottom":
        files = ["a", "b", "c", "d", "e", "f", "g", "h"]
        ranks = ["1", "2", "3", "4", "5", "6", "7", "8"]
        return f"{files[col]}{ranks[row]}"

    elif orientation == "top":
        files = ["h", "g", "f", "e", "d", "c", "b", "a"]
        ranks = ["8", "7", "6", "5", "4", "3", "2", "1"]
        return f"{files[col]}{ranks[row]}"

    elif orientation == "left":
        ranks = ["1", "2", "3", "4", "5", "6", "7", "8"]
        files = ["a", "b", "c", "d", "e", "f", "g", "h"]
        return f"{ranks[col]}{files[row]}"

    elif orientation == "right":
        ranks = ["8", "7", "6", "5", "4", "3", "2", "1"]
        files = ["h", "g", "f", "e", "d", "c", "b", "a"]
        return f"{ranks[col]}{files[row]}"

    else:
        raise ValueError(f"Invalid orientation: {orientation}")


# Rotate board to canonical white bottom orientation for FEN
def to_fen_board_coords(board):
    board = np.array(board)
    return np.flipud(board).tolist()


def generate_full_fen(
    board,
    orientation,
    active_player="w",
    en_passant="-",
    halfmove=0,
    fullmove=1,
    interactive=False,
):
    # Build piece placement
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
    piece_placement = "/".join(fen_rows)

    # Castling rights detection
    castling = ""

    def piece_at(rank, file, expected):
        file_idx = ord(file) - ord("a")
        rank_idx = 8 - int(rank)
        return board[rank_idx][file_idx] == expected

    if piece_at("1", "e", "K"):
        if piece_at("1", "h", "R"):
            castling += "K"
        if piece_at("1", "a", "R"):
            castling += "Q"
    if piece_at("8", "e", "k"):
        if piece_at("8", "h", "r"):
            castling += "k"
        if piece_at("8", "a", "r"):
            castling += "q"
    if castling == "":
        castling = "-"

    # Interactive prompt
    if interactive:
        user_input = input("Active player (w/b) [default w]: ").strip().lower()
        if user_input in ["w", "b"]:
            active_player = user_input

        user_input = input("En passant target square [default -]: ").strip().lower()
        if user_input:
            en_passant = user_input

        user_input = input("Halfmove clock [default 0]: ").strip()
        if user_input.isdigit():
            halfmove = user_input

        user_input = input("Fullmove number [default 1]: ").strip()
        if user_input.isdigit():
            fullmove = user_input

    fen = f"{piece_placement} {active_player} {castling} {en_passant} {halfmove} {fullmove}"
    return fen


# ===== FEN Generation =====
def detections_to_fen(detections, orientation):
    board = [["" for _ in range(8)] for _ in range(8)]

    yolo_to_fen = {
        "white-pawn": "P",
        "white-rook": "R",
        "white-knight": "N",
        "white-bishop": "B",
        "white-queen": "Q",
        "white-king": "K",
        "black-pawn": "p",
        "black-rook": "r",
        "black-knight": "n",
        "black-bishop": "b",
        "black-queen": "q",
        "black-king": "k",
    }

    for det in detections:
        row, col = det["row"], det["col"]
        label = det["label"]
        piece = yolo_to_fen.get(label.lower(), "")
        board[row][col] = piece

    canonical_board = to_fen_board_coords(board)

    return canonical_board, "/".join(
        [
            "".join(str(len(list(g))) if k == "" else k for k, g in groupby(row))
            for row in canonical_board
        ]
    )


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
    H = compute_homography(TRAPEZIUM_CORNERS)
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
            col = min(max(int(grid_x), 0), 7)
            row = min(max(int(grid_y), 0), 7)

            row, col = rotate_grid(row, col, orientation)

            detections.append(
                {
                    "row": row,
                    "col": col,
                    "grid": (row, col),
                    "center": (cx.item(), cy.item()),
                    "label": result.names[int(cls)],
                    "confidence": conf.item(),
                }
            )

    return ChessDetectionResult(processed_img, detections, H, orientation)


# Example Usage:
# result = detect_chess_board("board.jpg", "left")
# print(result.fen)
# result.visualize()
