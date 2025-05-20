from flask import Flask, Request, Response
from PIL import Image
from io import BytesIO
from ultralytics import YOLO
from chess_detector import detect_chess_board
import os

app = Flask(__name__)

# ===== Load Model from Environment Variable =====
model_path = os.getenv("MODEL_PATH")

if not model_path or not os.path.isfile(model_path):
    raise RuntimeError("Missing or invalid MODEL_PATH environment variable.")

app.config["MODEL_PATH"] = model_path
app.config["MODEL"] = YOLO(model_path)


# ===== Endpoint: Parse Board and Return FEN =====
@app.post("/parse-board")
def parse_board(req: Request) -> Response:
    if req.mimetype not in ("image/png", "image/jpg", "image/jpeg"):
        return Response(
            f"Unexpected MIME type {req.mimetype}; Expected PNG or JPG", 400
        )

    if "orientation" not in req.args:
        return Response("Expected a board orientation parameter.", 400)

    try:
        result = detect_chess_board(
            app.config["MODEL"],
            Image.open(BytesIO(req.data)),
            orientation=req.args["orientation"],
        )
        return Response(result.fen, 200)
    except Exception as e:
        return Response(f"Error processing image: {e}", 500)
