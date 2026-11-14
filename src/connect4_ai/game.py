"""Pure, immutable Connect Four game rules.

The bottom row has index 0. A board stores cells row-by-row, and player values
are +1 and -1. Immutability makes states safe dictionary keys for search and RL.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

ROWS = 6
COLS = 7
CONNECT = 4
EMPTY = 0
PLAYER_ONE = 1
PLAYER_TWO = -1


@dataclass(frozen=True, slots=True)
class Board:
    cells: tuple[int, ...] = (EMPTY,) * (ROWS * COLS)

    def __post_init__(self) -> None:
        if len(self.cells) != ROWS * COLS:
            raise ValueError(f"board requires {ROWS * COLS} cells")
        if any(cell not in (EMPTY, PLAYER_ONE, PLAYER_TWO) for cell in self.cells):
            raise ValueError("cells must be -1, 0, or 1")

    @staticmethod
    def from_rows(rows_top_to_bottom: Iterable[Iterable[int]]) -> "Board":
        rows = [tuple(row) for row in rows_top_to_bottom]
        if len(rows) != ROWS or any(len(row) != COLS for row in rows):
            raise ValueError(f"expected {ROWS} rows of {COLS} cells")
        return Board(tuple(cell for row in reversed(rows) for cell in row))

    def at(self, row: int, col: int) -> int:
        return self.cells[row * COLS + col]

    def legal_moves(self) -> tuple[int, ...]:
        return tuple(col for col in range(COLS) if self.at(ROWS - 1, col) == EMPTY)

    def drop(self, column: int, player: int) -> "Board":
        if player not in (PLAYER_ONE, PLAYER_TWO):
            raise ValueError("player must be 1 or -1")
        if not 0 <= column < COLS:
            raise ValueError(f"column must be between 0 and {COLS - 1}")
        for row in range(ROWS):
            index = row * COLS + column
            if self.cells[index] == EMPTY:
                updated = list(self.cells)
                updated[index] = player
                return Board(tuple(updated))
        raise ValueError(f"column {column} is full")

    def winner(self) -> int:
        directions = ((1, 0), (0, 1), (1, 1), (1, -1))
        for row in range(ROWS):
            for col in range(COLS):
                player = self.at(row, col)
                if player == EMPTY:
                    continue
                for d_row, d_col in directions:
                    end_row = row + (CONNECT - 1) * d_row
                    end_col = col + (CONNECT - 1) * d_col
                    if not (0 <= end_row < ROWS and 0 <= end_col < COLS):
                        continue
                    if all(
                        self.at(row + step * d_row, col + step * d_col) == player
                        for step in range(1, CONNECT)
                    ):
                        return player
        return EMPTY

    def is_full(self) -> bool:
        return all(self.at(ROWS - 1, col) != EMPTY for col in range(COLS))

    def is_terminal(self) -> bool:
        return self.winner() != EMPTY or self.is_full()

    def mirrored(self) -> "Board":
        return Board(
            tuple(
                self.at(row, COLS - 1 - col)
                for row in range(ROWS)
                for col in range(COLS)
            )
        )

    def normalized(self, player: int) -> tuple[int, ...]:
        """Return cells from ``player``'s perspective (+1 means self)."""
        if player not in (PLAYER_ONE, PLAYER_TWO):
            raise ValueError("player must be 1 or -1")
        return tuple(cell * player for cell in self.cells)

    def render(self) -> str:
        symbols = {EMPTY: ".", PLAYER_ONE: "X", PLAYER_TWO: "O"}
        lines = [
            " ".join(symbols[self.at(row, col)] for col in range(COLS))
            for row in range(ROWS - 1, -1, -1)
        ]
        lines.append(" ".join(str(col + 1) for col in range(COLS)))
        return "\n".join(lines)

