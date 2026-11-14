# Connect Four: Search vs. Reinforcement Learning

## Research question

How do alpha-beta search and self-play Q-learning compare on Connect Four, and
how do the Q-learning rate and epsilon-decay schedule affect win rate and
learning stability?

## Included

- A tested 6-by-7 Connect Four environment
- Random, tactical, alpha-beta, and Q-learning agents
- Self-play training with configurable learning parameters
- A seeded experiment runner that writes raw CSV data and a JSON summary
- An optional plot generator for learning curves and evaluation results
- A human-vs-agent command-line game

## Verify

```powershell
python -m unittest discover -s tests -v
```

## Quick start

Train one Q-table:

```powershell
python -m connect4_ai.train --episodes 50000 --seed 7 --output models/q_agent.json
```

Play against it (columns are numbered 1 through 7):

```powershell
python -m connect4_ai.play --agent q --model models/q_agent.json
```

Play against alpha-beta search:

```powershell
python -m connect4_ai.play --agent alpha-beta --depth 4
```

Run the planned comparison (start with `--quick` as a smoke test):

```powershell
python -m connect4_ai.experiment --quick
python -m connect4_ai.experiment --episodes 50000 --seeds 7 21 42
```

Outputs are written to a timestamped directory under `results/`. Each run saves
its configuration, episode-level training metrics, evaluation games, aggregate
statistics, trained Q-tables, and plots when Matplotlib is installed.

## Experimental design

The default full run varies:

- learning rate: `0.05`, `0.10`, `0.20`
- epsilon-decay fraction: `0.25`, `0.60`, `1.00`
- random seeds: `7`, `21`, `42`

Every trained model is evaluated from both player positions against random,
tactical, and depth-2 alpha-beta baselines. The primary metric is win rate;
draw rate, loss rate, average moves, training time, and Q-table size are also
recorded. Alpha-beta depth is deliberately modest in batch evaluation so the
experiment remains feasible on limited department hardware.
