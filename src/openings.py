from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Opening:
    eco_idx: str
    name: str
    moves: list[tuple[str, Optional[str]]]


def process_movelist(moves: str) -> list[tuple[str, Optional[str]]]:
    movelist = map(lambda block: block.strip().split(), moves.split(".")[1:])
    return [(block[0], block[1] if len(block) > 1 else None) for block in movelist]


def load_openings(path: str) -> dict[str, Opening]:
    openings = {}

    with open(path, "r") as stream:
        for line in stream.readlines():
            eco_idx, name, movelist_str, _, fen = line.strip().split("\t")
            openings[fen] = Opening(eco_idx, name, process_movelist(movelist_str))

    return openings
