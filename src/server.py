from flask import Flask, Request, Response
from PIL import Image
from io import BytesIO

app = Flask(__name__)


@app.post("/parse-board")
def parse_board(req: Request) -> Response:
    if req.mimetype not in ("image/png", "image/jpg"):
        return Response(
            f"Unexpected MIME type {req.mimetype}; Expected PNG or JPG", 400
        )

    image = Image.open(BytesIO(req.data))
    ...  # TODO call detect_chess_board with Image
    return Response("Not implemented", 501)
