"""Command-line entry point for training a single self-play model."""

from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path

from .agents import QLearningAgent
from .training import train_self_play


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=50_000)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--decay-fraction", type=float, default=0.6)
    parser.add_argument("--log-every", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("models/q_agent.json"))
    parser.add_argument("--metrics", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    agent = QLearningAgent(
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon_start,
        rng=random.Random(args.seed),
    )
    started = time.perf_counter()
    metrics = train_self_play(
        agent,
        args.episodes,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        decay_fraction=args.decay_fraction,
        log_every=args.log_every,
    )
    elapsed = time.perf_counter() - started
    metadata = {
        "episodes": args.episodes,
        "seed": args.seed,
        "epsilon_start": args.epsilon_start,
        "epsilon_end": args.epsilon_end,
        "decay_fraction": args.decay_fraction,
        "training_seconds": elapsed,
    }
    agent.save(args.output, metadata)
    metrics_path = args.metrics or args.output.with_suffix(".metrics.csv")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0].as_dict()))
        writer.writeheader()
        writer.writerows(metric.as_dict() for metric in metrics)
    print(f"Saved {len(agent.q):,} Q-values to {args.output}")
    print(f"Saved training metrics to {metrics_path}")
    print(f"Training time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()

