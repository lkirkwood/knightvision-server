import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
from chess_detector import map_point_to_board_space

# ===== 1. Original vs Corrected Image Comparison =====
def compare_images(img1: Image.Image, img2: Image.Image, label1="Original", label2="Corrected"):
    img1 = img1.resize((512, 512))
    img2 = img2.resize((512, 512))
    combined = Image.new("RGB", (1024, 512))
    combined.paste(img1, (0, 0))
    combined.paste(img2, (512, 0))

    plt.figure(figsize=(10, 5))
    plt.imshow(combined)
    plt.title(f"{label1} (Left)  |  {label2} (Right)")
    plt.axis("off")
    plt.show()

# ===== 2. Bounding Box Comparison =====
def map_bounding_boxes(image, detections, title, box_color="cyan"):
    image = image.copy()
    draw = ImageDraw.Draw(image)
    
    def get_square_name(row, col):
        files = "abcdefgh"
        ranks = "12345678"
        return f"{files[col]}{ranks[7 - row]}"  # standard white-bottom FEN

    for det in detections:
        # Draw bounding box
        if "box" in det:
            x1, y1, x2, y2 = det["box"]
            draw.rectangle([x1, y1, x2, y2], outline=box_color, width=2)
        
        # Draw center point and grid label if both are available
        if "center" in det and "grid" in det and det["grid"] is not None:
            cx, cy = det["center"]
            row_col = det["grid"]
            if isinstance(row_col, tuple) and len(row_col) == 2:
                row, col = row_col
                draw.ellipse((cx - 3, cy - 3, cx + 3, cy + 3), outline="red", width=2)
                square_name = get_square_name(row, col)
                draw.text((cx + 4, cy - 12), square_name, fill="yellow")

    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.title(title)
    plt.axis("off")
    plt.show()

# ===== 3. Projected Corners Back to Original =====
def map_corner_projection(original_image, corrected_image, homography_matrix):
    corrected_w, corrected_h = corrected_image.size
    corrected_corners = [(0, 0), (corrected_w, 0), (corrected_w, corrected_h), (0, corrected_h)]

    H_inv = np.linalg.inv(homography_matrix)
    def project_back(pt):
        vec = np.array([pt[0], pt[1], 1.0]).reshape(3, 1)
        proj = H_inv @ vec
        proj /= proj[2]
        return proj[0][0], proj[1][0]

    projected = [project_back(pt) for pt in corrected_corners]
    draw_img = original_image.copy()
    draw = ImageDraw.Draw(draw_img)
    for x, y in projected:
        draw.ellipse([x-4, y-4, x+4, y+4], outline="red", width=2)
    for i in range(4):
        x1, y1 = projected[i]
        x2, y2 = projected[(i+1)%4]
        draw.line([x1, y1, x2, y2], fill="yellow", width=2)

    combined = Image.new("RGB", (1024, 512))
    combined.paste(draw_img.resize((512, 512)), (0, 0))
    combined.paste(corrected_image.resize((512, 512)), (512, 0))

    plt.figure(figsize=(10, 5))
    plt.imshow(combined)
    plt.title("Projected Corners on Original | Corrected Image")
    plt.axis("off")
    plt.show()

# ===== 4. FEN-Based Digital Board Visualisation =====
def display_FEN(fen_string):
    square_size = 64
    board_img = Image.new("RGB", (512, 512), "white")
    draw = ImageDraw.Draw(board_img)
    font = ImageFont.load_default()

    # Draw 8x8 board
    for row in range(8):
        for col in range(8):
            fill = "#EEE" if (row + col) % 2 == 0 else "#555"
            draw.rectangle([col*square_size, row*square_size, (col+1)*square_size, (row+1)*square_size], fill=fill)

    # Fill in pieces
    fen_parts = fen_string.split()
    rows = fen_parts[0].split('/')
    for r_idx, fen_row in enumerate(rows):
        c_idx = 0
        for char in fen_row:
            if char.isdigit():
                c_idx += int(char)
            else:
                x = c_idx * square_size + 20
                y = r_idx * square_size + 20
                draw.text((x, y), char, fill="yellow", font=font)
                c_idx += 1

    plt.figure(figsize=(6, 6))
    plt.imshow(board_img)
    plt.title("FEN-Based Board Visualisation")
    plt.axis("off")
    plt.show()
