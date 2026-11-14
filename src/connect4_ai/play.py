"""Play Connect Four against one of the project agents."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from .agents import AlphaBetaAgent, QLearningAgent, RandomAgent, TacticalAgent
from .game import Board, EMPTY, PLAYER_ONE, PLAYER_TWO


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--agent",
        choices=("alpha-beta", "q", "random", "tactical"),
        default="alpha-beta",
    )
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--model", type=Path, default=Path("models/q_agent.json"))
    parser.add_argument("--human-second", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    return parser


def _make_agent(args: argparse.Namespace):
    if args.agent == "alpha-beta":
        return AlphaBetaAgent(args.depth)
    if args.agent == "q":
        if not args.model.exists():
            raise SystemExit(f"Q-table not found: {args.model}. Train it first.")
        return QLearningAgent.load(args.model, args.seed)
    if args.agent == "tactical":
        return TacticalAgent(random.Random(args.seed))
    return RandomAgent(random.Random(args.seed))


def _human_action(board: Board) -> int:
    while True:
        raw = input("Choose a column (1-7): ").strip()
        try:
            column = int(raw) - 1
        except ValueError:
            print("Enter a whole number from 1 through 7.")
            continue
        if column in board.legal_moves():
            return column
        print("That column is unavailable.")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    computer = _make_agent(args)
    human = PLAYER_TWO if args.human_second else PLAYER_ONE
    board, player = Board(), PLAYER_ONE
    print("You are", "O" if human == PLAYER_TWO else "X")
    while not board.is_terminal():
        print(f"\n{board.render()}\n")
        if player == human:
            action = _human_action(board)
        else:
            action = computer.choose_action(board, player)
            print(f"Computer chooses column {action + 1}.")
        board = board.drop(action, player)
        player = -player
    print(f"\n{board.render()}\n")
    winner = board.winner()
    if winner == EMPTY:
        print("Draw.")
    elif winner == human:
        print("You win!")
    else:
        print("Computer wins.")


if __name__ == "__main__":
    main()

