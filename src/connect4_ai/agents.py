"""Agents used as baselines and learning methods."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .game import Board, COLS, CONNECT, EMPTY, PLAYER_ONE, PLAYER_TWO, ROWS


class Agent(Protocol):
    def choose_action(self, board: Board, player: int) -> int: ...


def ordered_moves(board: Board) -> tuple[int, ...]:
    """Center-first ordering improves alpha-beta pruning."""
    center = COLS // 2
    return tuple(sorted(board.legal_moves(), key=lambda col: (abs(col - center), col)))


@dataclass(slots=True)
class RandomAgent:
    rng: random.Random = field(default_factory=random.Random)

    def choose_action(self, board: Board, player: int) -> int:
        del player
        legal = board.legal_moves()
        if not legal:
            raise ValueError("no legal moves")
        return self.rng.choice(legal)


@dataclass(slots=True)
class TacticalAgent:
    """Win immediately, block an immediate loss, otherwise play randomly."""

    rng: random.Random = field(default_factory=random.Random)

    def choose_action(self, board: Board, player: int) -> int:
        legal = board.legal_moves()
        if not legal:
            raise ValueError("no legal moves")
        for column in ordered_moves(board):
            if board.drop(column, player).winner() == player:
                return column
        for column in ordered_moves(board):
            if board.drop(column, -player).winner() == -player:
                return column
        return self.rng.choice(legal)


@dataclass(slots=True)
class AlphaBetaAgent:
    depth: int = 4
    nodes_searched: int = field(default=0, init=False)

    def choose_action(self, board: Board, player: int) -> int:
        if self.depth < 1:
            raise ValueError("depth must be at least 1")
        if board.is_terminal():
            raise ValueError("cannot choose an action in a terminal state")
        self.nodes_searched = 0
        best_score = -math.inf
        best_move = ordered_moves(board)[0]
        alpha, beta = -math.inf, math.inf
        for column in ordered_moves(board):
            child = board.drop(column, player)
            score = self._search(child, -player, player, self.depth - 1, alpha, beta)
            if score > best_score:
                best_score, best_move = score, column
            alpha = max(alpha, best_score)
        return best_move

    def _search(
        self,
        board: Board,
        turn: int,
        root_player: int,
        depth: int,
        alpha: float,
        beta: float,
    ) -> float:
        self.nodes_searched += 1
        winner = board.winner()
        if winner == root_player:
            return 1_000_000 + depth
        if winner == -root_player:
            return -1_000_000 - depth
        if board.is_full():
            return 0
        if depth == 0:
            return float(_heuristic(board, root_player))

        maximizing = turn == root_player
        value = -math.inf if maximizing else math.inf
        for column in ordered_moves(board):
            child = board.drop(column, turn)
            score = self._search(child, -turn, root_player, depth - 1, alpha, beta)
            if maximizing:
                value = max(value, score)
                alpha = max(alpha, value)
            else:
                value = min(value, score)
                beta = min(beta, value)
            if alpha >= beta:
                break
        return value


def _heuristic(board: Board, player: int) -> int:
    score = 3 * sum(board.at(row, COLS // 2) == player for row in range(ROWS))
    windows: list[tuple[int, ...]] = []
    for row in range(ROWS):
        for col in range(COLS - CONNECT + 1):
            windows.append(tuple(board.at(row, col + step) for step in range(CONNECT)))
    for row in range(ROWS - CONNECT + 1):
        for col in range(COLS):
            windows.append(tuple(board.at(row + step, col) for step in range(CONNECT)))
    for row in range(ROWS - CONNECT + 1):
        for col in range(COLS - CONNECT + 1):
            windows.append(tuple(board.at(row + step, col + step) for step in range(CONNECT)))
            windows.append(
                tuple(board.at(row + CONNECT - 1 - step, col + step) for step in range(CONNECT))
            )
    weights = {1: 1, 2: 10, 3: 80}
    for window in windows:
        own = window.count(player)
        opponent = window.count(-player)
        if opponent == 0:
            score += weights.get(own, 0)
        if own == 0:
            score -= weights.get(opponent, 0)
    return score


@dataclass(slots=True)
class QLearningAgent:
    alpha: float = 0.1
    gamma: float = 0.95
    epsilon: float = 0.1
    rng: random.Random = field(default_factory=random.Random)
    q: dict[str, float] = field(default_factory=dict)

    def _state(self, board: Board, player: int) -> tuple[str, bool]:
        normal = board.normalized(player)
        mirrored = board.mirrored().normalized(player)
        use_mirror = mirrored < normal
        canonical = mirrored if use_mirror else normal
        return "".join(str(value + 1) for value in canonical), use_mirror

    def _key(self, board: Board, player: int, action: int) -> str:
        state, mirrored = self._state(board, player)
        canonical_action = COLS - 1 - action if mirrored else action
        return f"{state}:{canonical_action}"

    def value(self, board: Board, player: int, action: int) -> float:
        return self.q.get(self._key(board, player, action), 0.0)

    def choose_action(self, board: Board, player: int, explore: bool = False) -> int:
        legal = board.legal_moves()
        if not legal:
            raise ValueError("no legal moves")
        if explore and self.rng.random() < self.epsilon:
            return self.rng.choice(legal)
        values = {action: self.value(board, player, action) for action in legal}
        best_value = max(values.values())
        best = [action for action, value in values.items() if value == best_value]
        return self.rng.choice(best)

    def update(
        self,
        board: Board,
        player: int,
        action: int,
        reward: float,
        next_board: Board | None = None,
    ) -> None:
        key = self._key(board, player, action)
        current = self.q.get(key, 0.0)
        if next_board is None or next_board.is_terminal():
            target = reward
        else:
            next_values = [self.value(next_board, player, move) for move in next_board.legal_moves()]
            target = reward + self.gamma * max(next_values, default=0.0)
        self.q[key] = current + self.alpha * (target - current)

    def save(self, path: str | Path, metadata: dict[str, object] | None = None) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": 1,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "metadata": metadata or {},
            "q": self.q,
        }
        destination.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, seed: int | None = None) -> "QLearningAgent":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format_version") != 1:
            raise ValueError("unsupported Q-table format")
        return cls(
            alpha=float(payload["alpha"]),
            gamma=float(payload["gamma"]),
            epsilon=float(payload["epsilon"]),
            rng=random.Random(seed),
            q={key: float(value) for key, value in payload["q"].items()},
        )

