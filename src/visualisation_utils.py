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
    for det in detections:
        # Draw bounding box
        if "box" in det:
            x1, y1, x2, y2 = det["box"]
            draw.rectangle([x1, y1, x2, y2], outline=box_color, width=2)
        
        # Draw center point and grid label if both are available
        if "center" in det and "grid" in det:
            cx, cy = det["center"]
            draw.ellipse((cx - 3, cy - 3, cx + 3, cy + 3), outline="red", width=2)

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