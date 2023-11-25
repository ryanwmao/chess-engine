import board
import numpy as np

def test_conversions():
  for ch in [chr(i) for i in range(97, 105)]:
    for j in range(1, 9):
      # print(ch, j)
      assert(board.int_to_square(board.square_to_int(ch, j)) == (ch, j))


if __name__ == "__main__":
  print("Tests:")
  # test_conversions()
  game = board.Game()
  game.pretty_print()
  # print(game.cpu.possible_moves(game.player))
  print(game.minimax(game.player, game.cpu, depth=3))
  print()
  game.player.pawn = np.uint64(1 << 40)
  game.pretty_print()
  print(game.minimax(game.player, game.cpu, depth=3))
  # print(game.cpu.possible_moves(game.player))