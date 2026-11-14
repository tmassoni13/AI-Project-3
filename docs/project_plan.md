# Project Plan: Connect Four Search vs. Reinforcement Learning

## Assignment option and scope

This project uses Option 1.3, **Adversarial Game Agent**. We implement Connect
Four and compare alpha-beta search with a strategy learned by tabular Q-learning
and self-play. Random and one-ply tactical agents provide additional baselines.
The scope is suitable for limited computing resources: the board and algorithms
are implemented locally, the learned model is a compact dictionary, and batch
search uses a controlled depth.

## Research questions

1. How does self-play Q-learning compare with random, tactical, and alpha-beta
   opponents when evaluated from both first- and second-player positions?
2. How do learning rate and epsilon-decay duration affect final win rate,
   training stability, and the number of learned state-action values?
3. What failure modes remain for a tabular learner in Connect Four's large state
   space?

## Hypotheses (recorded before the full experiment)

- Q-learning should substantially outperform a random policy after training.
- Slower exploration decay should improve generalization against tactical and
  search opponents, but may take longer to stabilize.
- Alpha-beta should remain stronger than the tabular learner because it can
  reason about states that self-play has not visited.
- Playing first will confer an advantage, so position-balanced evaluation is
  necessary.

These are hypotheses, not results. Retain, reject, or qualify each one using the
final data.

## Methods

### Environment

- Standard 6-row by 7-column Connect Four board
- Players alternate dropping pieces into non-full columns
- Rewards for the learner: `+1` win, `-1` loss, `0` draw/nonterminal transition
- Horizontally mirrored states share Q-values to reduce redundant learning

### Alpha-beta baseline

The search agent uses minimax with alpha-beta pruning, center-first move ordering,
and a heuristic that rewards center control and unblocked windows of two or
three pieces. Terminal outcomes dominate heuristic values. Search depth is
reported with every result.

### Q-learning agent

One Q-table is trained from both players' perspectives through self-play. A
player's transition includes the opponent's intervening move, so bootstrapping
occurs when the same player next acts. Action selection is epsilon-greedy during
training and greedy during evaluation.

Update rule:

`Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]`

Terminal transitions omit the future-value term.

## Experiment matrix

| Variable | Planned values |
|---|---|
| Learning rate (`alpha`) | 0.05, 0.10, 0.20 |
| Discount (`gamma`) | 0.95 (controlled) |
| Epsilon | 1.00 to 0.05, linear schedule |
| Decay fraction | 0.25, 0.60, 1.00 of training |
| Seeds | 7, 21, 42 |
| Training episodes | 50,000 initially; increase only if justified by curves |
| Evaluation opponents | random, tactical, alpha-beta depth 2 |
| Evaluation games | 200 per matchup, alternating player position |

Run `python -m connect4_ai.experiment --quick` first. After verification, run the
full matrix. Do not tune parameters on the final evaluation games