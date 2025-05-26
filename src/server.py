from typing import Any
from flask import Flask, Request, Response, request
from PIL import Image, ExifTags
from ultralytics import YOLO
from chess_detector import detect_chess_board
import os
import subprocess
import numpy as np
from tempfile import TemporaryDirectory
from io import BytesIO
import json

from openings import load_openings

app = Flask(__name__)

# ===== Load Model from Environment Variable =====
model_path = os.getenv("MODEL_PATH")

if not model_path or not os.path.isfile(model_path):
    raise RuntimeError("Missing or invalid MODEL_PATH environment variable.")

board_iso_path = os.getenv("BOARD_ISO_EXE")

if not board_iso_path or not os.path.isfile(board_iso_path):
    raise RuntimeError("Missing or invalid BOARD_ISO_EXE environment variable.")

openings_book = os.getenv("OPENINGS_PATH")

if not openings_book or not os.path.isfile(openings_book):
    raise RuntimeError("Missing or invalid OPENINGS_PATH environment variable.")


app.config["MODEL_PATH"] = model_path
app.config["MODEL"] = YOLO(model_path)
app.config["BOARD_ISO_EXE"] = board_iso_path
app.config["OPENINGS"] = load_openings(openings_book)


def process_input_img(img: Image.Image) -> Image.Image:
    # Get orientation tag ID
    orientation_tag = None
    for tag in ExifTags.TAGS:
        if ExifTags.TAGS[tag] == "Orientation":
            orientation_tag = tag
            break

    if orientation_tag is not None:
        exif = img._getexif()  # type: ignore
        if exif is not None:
            orientation = exif.get(orientation_tag, None)
            if orientation == 3:
                img = img.rotate(180, expand=True)
            elif orientation == 6:
                img = img.rotate(270, expand=True)
            elif orientation == 8:
                img = img.rotate(90, expand=True)

    img_width, img_height = img.size
    print(f"height: {img_height}, width: {img_width}")
    top = (img_height / 2) - (img_width / 2)
    bottom = (img_height / 2) + (img_width / 2)
    print(f"top: {top}, bottom: {bottom}")
    img = img.crop((0, top, img_width, bottom)).resize((512, 512))

    return img


# ===== Endpoint: Parse Board and Return FEN =====
@app.post("/parse-board")
def parse_board() -> Response:
    req: Request = request
    match req.mimetype:
        case "image/png":
            input_img_fext = "png"
        case "image/jpg" | "image/jpeg":
            input_img_fext = "jpeg"
        case "image/webp":
            input_img_fext = "webp"
        case other:
            return Response(f"Unexpected MIME type {other}; Expected PNG or JPG", 400)

    if "orientation" in req.args:
        orientation = req.args["orientation"]
    else:
        # return Response("Expected a board orientation parameter.", 400)
        orientation = "left"

    input_img = process_input_img(Image.open(BytesIO(req.data)))
    input_img.save("/tmp/last-input.jpg")

    with TemporaryDirectory() as tmpdir:
        try:
            input_path = os.path.join(tmpdir, f"input.{input_img_fext}")
            input_img.save(input_path)

            output_img_path = os.path.join(tmpdir, "output.jpg")
            output_homog_path = os.path.join(tmpdir, "homography.npy")

            subprocess.check_call(
                [
                    app.config["BOARD_ISO_EXE"],
                    input_path,
                    output_img_path,
                    output_homog_path,
                ]
            )
            homography = np.load(output_homog_path)
        except Exception as exc:
            print(exc)
            return Response(f"Error processing image: {exc}", 500)

        try:
            result = detect_chess_board(
                app.config["MODEL"],
                input_img,
                homography_matrix=homography,
                orientation=orientation,
            )
            response: dict[str, Any] = {"fen": result.fen}
            opening = app.config["OPENINGS"].get(result.fen)
            if opening is not None:
                response["opening"] = {"name": opening.name, "moves": opening.moves}

            return Response(json.dumps(response), 200)
        except Exception as exc:
            print(exc)
            return Response(f"Error detecting board state: {exc}", 500)
