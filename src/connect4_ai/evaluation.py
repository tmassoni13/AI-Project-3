"""Fair, position-balanced agent evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .agents import Agent
from .game import Board, EMPTY, PLAYER_ONE, PLAYER_TWO


@dataclass(frozen=True, slots=True)
class GameResult:
    game: int
    subject_player: int
    winner: int
    outcome: str
    moves: int

    def as_dict(self) -> dict[str, int | str]:
        return asdict(self)


def play_game(player_one: Agent, player_two: Agent) -> tuple[int, int]:
    board = Board()
    player = PLAYER_ONE
    agents = {PLAYER_ONE: player_one, PLAYER_TWO: player_two}
    moves = 0
    while not board.is_terminal():
        action = agents[player].choose_action(board, player)
        if action not in board.legal_moves():
            raise ValueError(f"agent selected illegal column {action}")
        board = board.drop(action, player)
        moves += 1
        player = -player
    return board.winner(), moves


def evaluate(subject: Agent, opponent: Agent, games: int) -> list[GameResult]:
    if games < 2:
        raise ValueError("at least two games are required to balance positions")
    records: list[GameResult] = []
    for game in range(1, games + 1):
        subject_player = PLAYER_ONE if game % 2 else PLAYER_TWO
        if subject_player == PLAYER_ONE:
            winner, moves = play_game(subject, opponent)
        else:
            winner, moves = play_game(opponent, subject)
        if winner == EMPTY:
            outcome = "draw"
        elif winner == subject_player:
            outcome = "win"
        else:
            outcome = "loss"
        records.append(GameResult(game, subject_player, winner, outcome, moves))
    return records

