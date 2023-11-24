import numpy as np
from typing import Optional, Tuple, List

'''
square: chess board label for a cell (e.g. 'c3')
int: 64-bit integer corresponding to board state for a piece
'''

def square_to_int(row: str, col: int) -> Optional[int]:
  '''
  Convert a row, col pair into integer representation
  '''
  if len(row) == 1 and row >= 'a' and row <= 'h' and col >= 1 and col <= 8:
    return np.uint64(1 << ((col - 1) * 8 + (ord(row) - ord('a'))))

def int_to_square(num: int) -> Optional[Tuple[str, int]]:
  '''
  Convert an integer corresponding to 
  '''
  if isinstance(num, np.uint64) and bin(num).count("1") == 1:
    lshift = 0
    for i in range(0, 64):
      if num & np.uint64(1 << i):
        lshift = i
        break
    col = int(lshift / 8) + 1
    row = chr(lshift % 8 + ord('a'))
    return row, col

class Player:
  pawn, rook, knight, bishop, king, queen = 'p', 'r', 'h', 'b', 'k', 'q'
  w_icons = {
     'p': '♙', 
     'r': '♖', 
     'h': '♘', 
     'b': '♗', 
     'k': '♔', 
     'q': '♕'
  }
  b_icons = {
     'p': '♟', 
     'r': '♜', 
     'h': '♞', 
     'b': '♝', 
     'k': '♚', 
     'q': '♛'
  }

  def __init__(self, white: bool):
    self.pawn = np.uint64(0)
    self.rook = np.uint64(0)
    self.knight = np.uint64(0)
    self.bishop = np.uint64(0)
    self.king = np.uint64(0)
    self.queen = np.uint64(0)
    self.reset(white)

  def reset(self, white: bool) -> None:
    self.pawn = np.uint64(0)
    if white:
      for i in range(8):
        self.pawn = self.pawn | np.uint64(1 << (8 + i))
      self.rook = np.uint64(1) | np.uint64(1 << 7)
      self.knight = np.uint64(2) | np.uint64(1 << 6)
      self.bishop = np.uint64(4) | np.uint64(1 << 5)
      self.queen = np.uint64(8)
      self.king = np.uint64(16)
    else:
      for i in range(8):
        self.pawn = self.pawn | np.uint64(1 << (48 + i))
      self.rook = np.uint64(1 << 56) | np.uint64(1 << 63)
      self.knight = np.uint64(1 << 57) | np.uint64(1 << 62)
      self.bishop = np.uint64(1 << 58) | np.uint(1 << 61)
      self.queen = np.uint(1 << 59)
      self.king = np.uint(1 << 60)

  def piece_at_int(self, num: int) -> Optional[str]:
    num = np.uint64(num)
    if self.pawn & num: return Player.pawn
    if self.rook & num: return Player.rook
    if self.knight & num: return Player.knight
    if self.bishop & num: return Player.bishop
    if self.queen & num: return Player.queen
    if self.king & num: return Player.king
  


class Game:
    def __init__(self):
        self.player1 = Player(white=True)
        self.player2 = Player(white=False)

    def pretty_print(self):
        board = [['.' for _ in range(8)] for _ in range(8)]
        for i in range(8):
            for j in range(8):
                piece1 = self.player1.piece_at_int(square_to_int(chr(ord('a') + i), j + 1))
                piece2 = self.player2.piece_at_int(square_to_int(chr(ord('a') + i), j + 1))
                if piece1:
                    board[j][i] = 'w' + piece1
                elif piece2:
                    board[j][i] = 'b' + piece2

        print("    a   b   c   d   e   f   g   h")
        print("    -----------------------------")
        for i in range(8):
            print(f" {8 - i} ", end="")
            for j in range(8):
                piece = board[i][j]
                if piece[0] == 'w':
                    print(f" {Player.w_icons[piece[1]]} ", end=" ")  # White piece
                elif piece[0] == 'b':
                    print(f" {Player.b_icons[piece[1]]} ", end=" ")  # Black piece
                else:
                    print(" • ", end=" ")  # Empty square
            print(f" {8 - i}")
        print("    -----------------------------")
        print("    a   b   c   d   e   f   g   h")