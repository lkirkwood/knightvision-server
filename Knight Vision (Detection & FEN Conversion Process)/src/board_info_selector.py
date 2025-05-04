# ===== Knight Vision - Board Info Selector =====
# Simple script to select corners and orientation ONLY (no detection logic in this file)

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# ===== Select Corners =====
def select_corners(image_path):
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img_np)
    plt.title("Click EXACTLY 4 board corners in order: Top-Left, Top-Right, Bottom-Right, Bottom-Left")

    points = plt.ginput(4, timeout=0)
    plt.close(fig)

    if len(points) != 4:
        raise ValueError("You must select exactly 4 corners.")

    return points

# ===== Select Orientation =====
def select_orientation(image_path):
    img = Image.open(image_path).convert("RGB")
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img)
    plt.title("Select which side White is starting from:\nClick Top / Bottom / Left / Right")

    orientation_result = {}

    def onclick(event):
        if event.xdata is None or event.ydata is None:
            return

        x, y = event.xdata, event.ydata
        w, h = img.size

        if y < h * 0.25:
            orientation_result['value'] = 'top'
        elif y > h * 0.75:
            orientation_result['value'] = 'bottom'
        elif x < w * 0.25:
            orientation_result['value'] = 'left'
        elif x > w * 0.75:
            orientation_result['value'] = 'right'
        plt.close()

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    return orientation_result.get('value', 'bottom')

# ===== Main flow (script usage) =====
if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python board_info_selector.py <image_path>")
        exit()

    image_path = sys.argv[1]

    corners = select_corners(image_path)
    orientation = select_orientation(image_path)

    print("\nCorners selected:")
    for pt in corners:
        print(f"[{pt[0]:.2f}, {pt[1]:.2f}]")

    print("\nOrientation selected:", orientation)

    print("\nCopy-paste ready format:")
    print("corners =", [list(p) for p in corners])
    print("orientation =", repr(orientation))
