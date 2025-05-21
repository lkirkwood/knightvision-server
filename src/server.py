from flask import Flask, Request, Response, request
from PIL import Image
from ultralytics import YOLO
from chess_detector import detect_chess_board
import os
import subprocess
import numpy as np
from tempfile import TemporaryDirectory
from io import BytesIO

app = Flask(__name__)

# ===== Load Model from Environment Variable =====
model_path = os.getenv("MODEL_PATH")

if not model_path or not os.path.isfile(model_path):
    raise RuntimeError("Missing or invalid MODEL_PATH environment variable.")

board_iso_path = os.getenv("BOARD_ISO_EXE")

if not board_iso_path or not os.path.isfile(board_iso_path):
    raise RuntimeError("Missing or invalid BOARD_ISO_EXE environment variable.")

app.config["MODEL_PATH"] = model_path
app.config["MODEL"] = YOLO(model_path)
app.config["BOARD_ISO_EXE"] = board_iso_path


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

    input_img = Image.open(BytesIO(req.data))
    input_img_height, input_img_width = input_img.size
    top = (input_img_height / 2) - (input_img_width / 2)
    bottom = (input_img_height / 2) + (input_img_width / 2)
    input_img = input_img.crop((0, top, input_img_width, bottom)).resize((512, 512))

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
            return Response(result.fen, 200)
        except Exception as exc:
            print(exc)
            return Response(f"Error detecting board state: {exc}", 500)
