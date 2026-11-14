import unittest

from connect4_ai.game import Board, PLAYER_ONE, PLAYER_TWO


class BoardTests(unittest.TestCase):
    def test_piece_falls_to_lowest_open_row(self):
        board = Board().drop(3, PLAYER_ONE).drop(3, PLAYER_TWO)
        self.assertEqual(board.at(0, 3), PLAYER_ONE)
        self.assertEqual(board.at(1, 3), PLAYER_TWO)

    def test_rejects_move_in_full_column(self):
        board = Board()
        for index in range(6):
            board = board.drop(0, PLAYER_ONE if index % 2 == 0 else PLAYER_TWO)
        with self.assertRaises(ValueError):
            board.drop(0, PLAYER_ONE)

    def test_finds_horizontal_vertical_and_diagonal_wins(self):
        horizontal = Board()
        for column in range(4):
            horizontal = horizontal.drop(column, PLAYER_ONE)
        self.assertEqual(horizontal.winner(), PLAYER_ONE)

        vertical = Board()
        for _ in range(4):
            vertical = vertical.drop(6, PLAYER_TWO)
        self.assertEqual(vertical.winner(), PLAYER_TWO)

        diagonal = Board.from_rows(
            [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 1, -1, 0, 0, 0],
                [0, 1, -1, -1, 0, 0, 0],
                [1, -1, -1, 1, 0, 0, 0],
            ]
        )
        self.assertEqual(diagonal.winner(), PLAYER_ONE)

    def test_mirroring_twice_restores_board(self):
        board = Board().drop(0, PLAYER_ONE).drop(5, PLAYER_TWO)
        self.assertEqual(board.mirrored().mirrored(), board)


if __name__ == "__main__":
    unittest.main()

