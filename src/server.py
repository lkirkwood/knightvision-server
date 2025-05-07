from flask import Flask, Request, Response
from PIL import Image
from io import BytesIO
from ultralytics import YOLO
from chess_detector import detect_chess_board
import os

app = Flask(__name__)

if not ("MODEL_PATH" in app.config and os.path.isfile(app.config["MODEL_FILE"])):
    raise RuntimeError("Missing MODEL_PATH configuration value.")

app.config["MODEL"] = YOLO(app.config["MODEL_PATH"])


@app.post("/parse-board")
def parse_board(req: Request) -> Response:
    if req.mimetype not in ("image/png", "image/jpg"):
        return Response(
            f"Unexpected MIME type {req.mimetype}; Expected PNG or JPG", 400
        )

    if "orientation" not in req.args:
        return Response("Expected a board orientation parameter.", 400)

    result = detect_chess_board(
        app.config["MODEL"], Image.open(BytesIO(req.data)), req.args["orientation"]
    )

    return Response(result.fen, 200)
