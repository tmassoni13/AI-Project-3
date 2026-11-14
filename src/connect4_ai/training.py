"""Self-play training logic shared by the CLI and experiment runner."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .agents import QLearningAgent
from .game import Board, EMPTY, PLAYER_ONE


@dataclass(frozen=True, slots=True)
class TrainingMetric:
    episode: int
    epsilon: float
    player_one_wins: int
    player_two_wins: int
    draws: int
    average_moves: float
    q_entries: int

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)


def epsilon_at(
    episode: int,
    episodes: int,
    start: float,
    end: float,
    decay_fraction: float,
) -> float:
    if not 0 < decay_fraction <= 1:
        raise ValueError("decay_fraction must be in (0, 1]")
    decay_episodes = max(1, int(episodes * decay_fraction))
    progress = min(1.0, episode / decay_episodes)
    return start + progress * (end - start)


def train_self_play(
    agent: QLearningAgent,
    episodes: int,
    *,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    decay_fraction: float = 0.6,
    log_every: int = 1000,
) -> list[TrainingMetric]:
    """Train one shared Q-table from both player perspectives.

    Each player's transition spans the opponent's intervening move. A terminal
    win gives +1 to the winning action and -1 to the losing player's pending
    action; a draw gives 0 to both. Nonterminal transitions bootstrap from the
    next state in which that same player is able to act.
    """
    if episodes < 1:
        raise ValueError("episodes must be positive")
    if log_every < 1:
        raise ValueError("log_every must be positive")

    metrics: list[TrainingMetric] = []
    window_p1 = window_p2 = window_draws = window_moves = window_games = 0

    for episode in range(1, episodes + 1):
        agent.epsilon = epsilon_at(
            episode - 1, episodes, epsilon_start, epsilon_end, decay_fraction
        )
        board = Board()
        player = PLAYER_ONE
        pending: dict[int, tuple[Board, int]] = {}
        moves = 0

        while True:
            # The opponent has now responded, so the current player's previous
            # action has reached its next decision state.
            previous = pending.pop(player, None)
            if previous is not None:
                previous_board, previous_action = previous
                agent.update(previous_board, player, previous_action, 0.0, board)

            action = agent.choose_action(board, player, explore=True)
            next_board = board.drop(action, player)
            moves += 1
            winner = next_board.winner()

            if winner != EMPTY:
                agent.update(board, player, action, 1.0, None)
                losing = pending.pop(-player, None)
                if losing is not None:
                    losing_board, losing_action = losing
                    agent.update(losing_board, -player, losing_action, -1.0, None)
                if winner == PLAYER_ONE:
                    window_p1 += 1
                else:
                    window_p2 += 1
                break

            if next_board.is_full():
                agent.update(board, player, action, 0.0, None)
                other = pending.pop(-player, None)
                if other is not None:
                    other_board, other_action = other
                    agent.update(other_board, -player, other_action, 0.0, None)
                window_draws += 1
                break

            pending[player] = (board, action)
            board = next_board
            player = -player

        window_moves += moves
        window_games += 1
        if episode % log_every == 0 or episode == episodes:
            metrics.append(
                TrainingMetric(
                    episode=episode,
                    epsilon=agent.epsilon,
                    player_one_wins=window_p1,
                    player_two_wins=window_p2,
                    draws=window_draws,
                    average_moves=window_moves / window_games,
                    q_entries=len(agent.q),
                )
            )
            window_p1 = window_p2 = window_draws = window_moves = window_games = 0

    agent.epsilon = epsilon_end
    return metrics

