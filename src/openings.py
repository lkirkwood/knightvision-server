from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Opening:
    eco_idx: str
    name: str
    moves: list[list[str]]


def process_movelist(moves: str) -> list[list[str]]:
    movelist = map(lambda block: block.strip().split(), moves.split(".")[1:])
    return [[block[i] for i in range(min(len(block), 2))] for block in movelist]


def load_openings(path: str) -> dict[str, Opening]:
    openings = {}

    with open(path, "r") as stream:
        for line in stream.readlines():
            eco_idx, name, movelist_str, _, fen = line.strip().split("\t")
            openings[fen] = Opening(eco_idx, name, process_movelist(movelist_str))

    return openings
