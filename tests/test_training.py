import random
import unittest

from connect4_ai.agents import QLearningAgent
from connect4_ai.training import epsilon_at, train_self_play


class TrainingTests(unittest.TestCase):
    def test_epsilon_schedule_reaches_floor(self):
        self.assertEqual(epsilon_at(0, 100, 1.0, 0.1, 0.5), 1.0)
        self.assertAlmostEqual(epsilon_at(50, 100, 1.0, 0.1, 0.5), 0.1)
        self.assertAlmostEqual(epsilon_at(99, 100, 1.0, 0.1, 0.5), 0.1)

    def test_short_training_run_creates_metrics_and_values(self):
        agent = QLearningAgent(rng=random.Random(4))
        metrics = train_self_play(agent, 10, log_every=5)
        self.assertEqual([metric.episode for metric in metrics], [5, 10])
        self.assertGreater(len(agent.q), 0)


if __name__ == "__main__":
    unittest.main()

