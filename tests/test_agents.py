import random
import tempfile
import unittest
from pathlib import Path

from connect4_ai.agents import AlphaBetaAgent, QLearningAgent, TacticalAgent
from connect4_ai.game import Board, PLAYER_ONE, PLAYER_TWO


class AgentTests(unittest.TestCase):
    def test_tactical_agent_takes_win(self):
        board = Board().drop(0, PLAYER_ONE).drop(1, PLAYER_ONE).drop(2, PLAYER_ONE)
        self.assertEqual(TacticalAgent(random.Random(1)).choose_action(board, PLAYER_ONE), 3)

    def test_alpha_beta_blocks_immediate_loss(self):
        board = Board().drop(0, PLAYER_TWO).drop(1, PLAYER_TWO).drop(2, PLAYER_TWO)
        self.assertEqual(AlphaBetaAgent(depth=2).choose_action(board, PLAYER_ONE), 3)

    def test_q_update_and_round_trip(self):
        board = Board()
        agent = QLearningAgent(alpha=0.5, rng=random.Random(1))
        agent.update(board, PLAYER_ONE, 3, 1.0)
        self.assertEqual(agent.value(board, PLAYER_ONE, 3), 0.5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "q.json"
            agent.save(path, {"test": True})
            restored = QLearningAgent.load(path, seed=1)
            self.assertEqual(restored.q, agent.q)

    def test_q_table_uses_horizontal_symmetry(self):
        board = Board().drop(0, PLAYER_ONE).drop(1, PLAYER_TWO)
        mirrored = board.mirrored()
        agent = QLearningAgent()
        agent.update(board, PLAYER_ONE, 2, 1.0)
        self.assertEqual(agent.value(mirrored, PLAYER_ONE, 4), agent.value(board, PLAYER_ONE, 2))


if __name__ == "__main__":
    unittest.main()

