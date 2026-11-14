"""Run the reproducible parameter and baseline comparison experiment."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
import statistics
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from .agents import AlphaBetaAgent, QLearningAgent, RandomAgent, TacticalAgent
from .evaluation import evaluate
from .training import train_self_play


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=50_000)
    parser.add_argument("--eval-games", type=int, default=200)
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[0.05, 0.1, 0.2])
    parser.add_argument("--decay-fractions", type=float, nargs="+", default=[0.25, 0.6, 1.0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 21, 42])
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--alpha-beta-depth", type=int, default=2)
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    parser.add_argument(
        "--quick",
        action="store_true",
        help="smoke test: one configuration, 1,000 episodes, 20 evaluation games",
    )
    return parser


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summarize(evaluation_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in evaluation_rows:
        groups.setdefault((str(row["run_id"]), str(row["opponent"])), []).append(row)
    summary: list[dict[str, object]] = []
    for (run_id, opponent), rows in sorted(groups.items()):
        counts = Counter(str(row["outcome"]) for row in rows)
        total = len(rows)
        first = rows[0]
        summary.append(
            {
                "run_id": run_id,
                "opponent": opponent,
                "alpha": first["alpha"],
                "decay_fraction": first["decay_fraction"],
                "seed": first["seed"],
                "games": total,
                "wins": counts["win"],
                "draws": counts["draw"],
                "losses": counts["loss"],
                "win_rate": counts["win"] / total,
                "draw_rate": counts["draw"] / total,
                "loss_rate": counts["loss"] / total,
                "average_moves": statistics.mean(float(row["moves"]) for row in rows),
            }
        )
    return summary


def _plot(run_dir: Path, training_rows: list[dict[str, object]], summary: list[dict[str, object]]) -> str:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return "Matplotlib not installed; CSV and JSON results were still created."

    figure, axis = plt.subplots(figsize=(9, 5))
    by_run: dict[str, list[dict[str, object]]] = {}
    for row in training_rows:
        by_run.setdefault(str(row["run_id"]), []).append(row)
    for run_id, rows in by_run.items():
        rows.sort(key=lambda item: int(item["episode"]))
        rates = []
        for row in rows:
            games = int(row["player_one_wins"]) + int(row["player_two_wins"]) + int(row["draws"])
            rates.append((int(row["player_one_wins"]) + int(row["player_two_wins"])) / games)
        axis.plot([int(row["episode"]) for row in rows], rates, alpha=0.65, label=run_id)
    axis.set(title="Self-play decisive-game rate", xlabel="Episode", ylabel="Win fraction in window")
    axis.set_ylim(0, 1.02)
    if len(by_run) <= 9:
        axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(run_dir / "learning_curves.png", dpi=160)
    plt.close(figure)

    labels = [f"{row['run_id']}\n{row['opponent']}" for row in summary]
    figure, axis = plt.subplots(figsize=(max(9, len(labels) * 0.55), 5))
    axis.bar(range(len(summary)), [float(row["win_rate"]) for row in summary])
    axis.set(title="Q-learning evaluation win rates", ylabel="Win rate", xticks=range(len(labels)))
    axis.set_xticklabels(labels, rotation=75, ha="right", fontsize=7)
    axis.set_ylim(0, 1)
    figure.tight_layout()
    figure.savefig(run_dir / "evaluation_win_rates.png", dpi=160)
    plt.close(figure)
    return "Created learning_curves.png and evaluation_win_rates.png."


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.quick:
        args.episodes = 1_000
        args.eval_games = 20
        args.learning_rates = [0.1]
        args.decay_fractions = [0.6]
        args.seeds = [args.seeds[0]]
    if args.eval_games < 2:
        raise SystemExit("--eval-games must be at least 2")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_root / f"run_{stamp}"
    models_dir = run_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=False)
    config = vars(args).copy()
    config["output_root"] = str(config["output_root"])
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    training_rows: list[dict[str, object]] = []
    evaluation_rows: list[dict[str, object]] = []
    timing_rows: list[dict[str, object]] = []
    combinations = list(itertools.product(args.learning_rates, args.decay_fractions, args.seeds))

    for index, (alpha, decay, seed) in enumerate(combinations, start=1):
        run_id = f"a{alpha:g}_d{decay:g}_s{seed}"
        print(f"[{index}/{len(combinations)}] Training {run_id}")
        agent = QLearningAgent(
            alpha=alpha,
            gamma=args.gamma,
            epsilon=args.epsilon_start,
            rng=random.Random(seed),
        )
        started = time.perf_counter()
        metrics = train_self_play(
            agent,
            args.episodes,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            decay_fraction=decay,
            log_every=max(1, args.episodes // 100),
        )
        elapsed = time.perf_counter() - started
        agent.save(
            models_dir / f"{run_id}.json",
            {"episodes": args.episodes, "seed": seed, "decay_fraction": decay},
        )
        for metric in metrics:
            training_rows.append(
                {
                    "run_id": run_id,
                    "alpha": alpha,
                    "decay_fraction": decay,
                    "seed": seed,
                    **metric.as_dict(),
                }
            )
        timing_rows.append(
            {
                "run_id": run_id,
                "alpha": alpha,
                "decay_fraction": decay,
                "seed": seed,
                "episodes": args.episodes,
                "training_seconds": elapsed,
                "q_entries": len(agent.q),
            }
        )

        opponents = {
            "random": RandomAgent(random.Random(seed + 1_000)),
            "tactical": TacticalAgent(random.Random(seed + 2_000)),
            f"alpha_beta_d{args.alpha_beta_depth}": AlphaBetaAgent(args.alpha_beta_depth),
        }
        for opponent_name, opponent in opponents.items():
            for record in evaluate(agent, opponent, args.eval_games):
                evaluation_rows.append(
                    {
                        "run_id": run_id,
                        "opponent": opponent_name,
                        "alpha": alpha,
                        "decay_fraction": decay,
                        "seed": seed,
                        **record.as_dict(),
                    }
                )

    summary = _summarize(evaluation_rows)
    _write_csv(run_dir / "training_metrics.csv", training_rows)
    _write_csv(run_dir / "evaluation_games.csv", evaluation_rows)
    _write_csv(run_dir / "evaluation_summary.csv", summary)
    _write_csv(run_dir / "training_timing.csv", timing_rows)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_message = _plot(run_dir, training_rows, summary)
    print(f"Results saved to {run_dir}")
    print(plot_message)


if __name__ == "__main__":
    main()

