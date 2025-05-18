import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2


def show_image(image: Image.Image, title: str = "Image"):
    """Displays a single image using matplotlib."""
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.title(title)
    plt.axis("off")
    plt.show()


def show_image_comparison(img1: Image.Image, img2: Image.Image, label1="Original", label2="Corrected"):
    """Displays two images side-by-side, optionally resizing them to match."""
    img1 = img1.resize((512, 512))
    img2 = img2.resize((512, 512))

    combined = Image.new("RGB", (1024, 512))
    combined.paste(img1, (0, 0))
    combined.paste(img2, (512, 0))

    plt.figure(figsize=(10, 5))
    plt.imshow(combined)
    plt.axis("off")
    plt.title(f"{label1} (Left)  |  {label2} (Right)")
    plt.show()


def draw_detections_on_image(image: Image.Image, detections: list, title="Detections"):
    """Draws bounding boxes, centerpoints, class labels, and confidence scores."""
    img = image.copy()
    draw = ImageDraw.Draw(img)

    for det in detections:
        cx, cy = det["center"]
        label = det["label"]
        confidence = det.get("confidence", None)
        box = det.get("box", None)  # [x1, y1, x2, y2] if available

        if box:
            x1, y1, x2, y2 = box
            draw.rectangle([x1, y1, x2, y2], outline="cyan", width=2)

        draw.ellipse((cx - 4, cy - 4, cx + 4, cy + 4), outline="red", width=2)
        label_text = f"{label}"
        if confidence is not None:
            label_text += f" ({confidence:.2f})"
        draw.text((cx + 6, cy - 12), label_text, fill="yellow")

    show_image(img, title)


def visualise_on_digital_board(detections: list, fen_background: Image.Image = None):
    """Overlays centerpoints and piece labels on a digital 8x8 chessboard."""
    if fen_background is None:
        board_img = Image.new("RGB", (512, 512), color="white")
        draw = ImageDraw.Draw(board_img)
        for r in range(8):
            for c in range(8):
                if (r + c) % 2 == 0:
                    draw.rectangle([
                        c * 64, r * 64, (c + 1) * 64, (r + 1) * 64
                    ], fill="#EEE")
                else:
                    draw.rectangle([
                        c * 64, r * 64, (c + 1) * 64, (r + 1) * 64
                    ], fill="#555")
    else:
        board_img = fen_background.copy().resize((512, 512))

    draw = ImageDraw.Draw(board_img)

    for det in detections:
        grid_row, grid_col = det["grid"]
        label = det["label"]
        confidence = det.get("confidence", None)

        center_x = grid_col * 64 + 32
        center_y = grid_row * 64 + 32

        draw.ellipse((center_x - 5, center_y - 5, center_x + 5, center_y + 5), outline="red", width=2)
        label_text = f"{label}"
        if confidence is not None:
            label_text += f" ({confidence:.2f})"
        draw.text((center_x + 6, center_y - 10), label_text, fill="yellow")

    show_image(board_img, "Detections on Grid-Aligned Chessboard")
