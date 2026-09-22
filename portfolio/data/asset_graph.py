"""Adjacency structure for the Fama-French 10x10 Size/Book-to-Market grid."""
import re

import numpy as np

_SPECIAL = {"SMALL LoBM": (1, 1), "SMALL HiBM": (1, 10), "BIG LoBM": (10, 1), "BIG HiBM": (10, 10)}
_PATTERN = re.compile(r"^ME(\d+) BM(\d+)$")


def _grid_position(name):
    if name in _SPECIAL:
        return _SPECIAL[name]
    match = _PATTERN.match(name)
    if not match:
        raise ValueError(f"Unrecognized Size-BM portfolio name: {name!r}")
    me, bm = int(match.group(1)), int(match.group(2))
    if not (1 <= me <= 10 and 1 <= bm <= 10):
        raise ValueError(f"Size-BM rank out of range: {name!r}")
    return me, bm


def build_size_bm_incidence(columns):
    """One signed row per adjacent Size-BM pair present in columns.

    Only horizontal (same book-to-market rank, adjacent size rank) and
    vertical (same size rank, adjacent book-to-market rank) edges between
    portfolios that are both present are included; no edge is drawn across a
    missing grid position (e.g. a dropped corner). Returns the (num_edges,
    num_assets) signed incidence matrix and the list of (i, j) index pairs it
    encodes, both in the order columns are given.
    """
    columns = list(columns)
    if len(set(columns)) != len(columns):
        raise ValueError("Asset names must be unique")
    position_to_index = {}
    for index, name in enumerate(columns):
        position = _grid_position(name)
        if position in position_to_index:
            raise ValueError(f"Duplicate grid position for {name!r}")
        position_to_index[position] = index
    edges = []
    for (me, bm), i in position_to_index.items():
        for neighbor in ((me + 1, bm), (me, bm + 1)):
            j = position_to_index.get(neighbor)
            if j is not None:
                edges.append((i, j))
    if not edges:
        raise ValueError("No adjacent Size-BM portfolios found")
    incidence = np.zeros((len(edges), len(columns)))
    for row, (i, j) in enumerate(edges):
        incidence[row, i] = 1.0
        incidence[row, j] = -1.0
    return incidence, edges
