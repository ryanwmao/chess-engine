import board

def test_conversions():
  for ch in [chr(i) for i in range(97, 105)]:
    for j in range(1, 9):
      # print(ch, j)
      assert(board.int_to_square(board.square_to_int(ch, j)) == (ch, j))


if __name__ == "__main__":
  print("Tests:")
  # test_conversions()
  game = board.Game()
  # game.pretty_print()
  # print(game.cpu.possible_moves(game.player))
  # print(game.minimax(game.player, game.cpu, depth=3))
  # print(game.cpu.sum_piece_bonuses())
  # print(game.player.sum_piece_bonuses())
  # print()
  
  # game.player.pawn = np.uint64(1 << 40)
  # game.player.bishop = 0
  # game.player.knight = 0
  # game.player.queen = 1 << 40
  # game.pretty_print()
  # print(game.minimax(game.player, game.cpu, depth=3))
  # print(game.cpu.possible_moves(game.player))
  # print(game.player.possible_moves(game.cpu))
  # print(game.player.can_castle_left, game.player.can_castle_right)
  # game.player, game.cpu = game.player.make_move((board.Player.king, 4, 2), game.cpu)
  # print(game.player.can_castle_left, game.player.can_castle_right)
  # game.pretty_print()
  game.cpu.knight = 0
  game.cpu.bishop = 0
  game.cpu.queen = 0
  # game.cpu, game.player = game.cpu.make_move((board.Player.king, 60, 58), game.player)
  # game.pretty_print()
  game.cpu.pawn = 0
  game.cpu.rook = 0
  game.player.rook = 0
  game.player.pawn = 0
  game.player.queen = 0
  game.player.bishop = 0
  game.player.knight = 0
  game.player.king = 0
  for i in range(64):
    game.cpu.king = 1 << i
    val = game.cpu.sum_piece_bonuses()
    if val >= 1000:
      print(i)
      print(game.cpu.sum_piece_bonuses())
      game.pretty_print()

  # game.cpu.king = 1
  # val = game.cpu.sum_piece_bonuses()
  # print(val)