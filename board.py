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
  
def square_to_leftshift_bits(row: str, col: int) -> Optional[int]:
   '''
   Convert a row, col pair into the number of left shift bits
   '''
   if len(row) == 1 and row >= 'a' and row <= 'h' and col >= 1 and col <= 8:
      return (col - 1) * 8 + (ord(row) - ord('a'))

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
    self.white = white
    self.last_move = None

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
    '''
    Returns a piece at a cell, if one exists.
    Input is the number of left shift bits.
    '''
    if num < 0 or num >= 64: return None
    num = np.uint64(1 << num)
    if self.pawn & num: return Player.pawn
    if self.rook & num: return Player.rook
    if self.knight & num: return Player.knight
    if self.bishop & num: return Player.bishop
    if self.queen & num: return Player.queen
    if self.king & num: return Player.king


  # we don't account for pins under the assumption that exposing the king
  # by moving the pin will be avoided through our reward function
  def possible_moves(self, other):
    def can_move(square):
       return other.piece_at_int(square) is None and self.piece_at_int(square) is None

    res = []
    for i in range(0, 64):
      cell = np.uint64(1 << i)
      col = i % 8
      row = int(i / 8)
      if self.pawn & cell:
        sign = 1 if self.white else -1
        # pawn taking diagonally
        take_left, take_right = i + 7 * sign, i + 9 * sign
        if other.piece_at_int(take_left): res.append((Player.pawn, i, take_left))
        if other.piece_at_int(take_right): res.append((Player.pawn, i, take_right))

        # pawn pushing forward, or pushing double from start
        move_forward, move_double = i + 8 * sign, i + 16 * sign
        if move_forward < 64 and move_forward >= 0:
            if can_move(move_forward):
              # promote pawn
              if (self.white and int(move_forward / 8) == 7) or (not self.white and int(move_forward / 8) == 0):
                 for piece in [Player.bishop, Player.knight, Player.rook, Player.queen]:
                  res.append((Player.pawn, i, move_forward, piece))
              else:
                 res.append((Player.pawn, i, move_forward))
            # push two spaces from start
            if ((int(i / 8) == 1 and self.white) or \
                (int(i / 8) == 6) and not self.white) and \
                can_move(move_double):
                  res.append((Player.pawn, i, move_double))
        
        # en passant
        if other.last_move and other.last_move[0] == Player.pawn \
          and abs(other.last_move[1] - other.last_move[2]) == 16:
            if other.last_move[2] == i - 1 * sign:
              res.append((Player.pawn, i, i + 7 * sign))
            if other.last_move[2] == i + 1 * sign:
              res.append((Player.pawn, i, i + 9 * sign))

      if self.rook & cell:
         # move left
         j = i - 1
         while j >= 0:
            if can_move(j): res.append((Player.rook, i, j))
            else:
               if self.piece_at_int(j) is None:
                  res.append((Player.rook, i, j))
               break
            if j % 8 == 0: break
            j -= 1

         # move right
         j = i + 1
         while j < 64 and j % 8 != 0:
            if can_move(j): res.append((Player.rook, i, j))
            else:
               if self.piece_at_int(j) is None:
                  res.append((Player.rook, i, j))
               break
         
         # move up
         j = i + 8
         while (j < 64):
            if can_move(j): res.append((Player.rook, i, j))
            else:
               if self.piece_at_int(j) is None:
                  res.append((Player.rook, i, j))
               break
            j += 8
         
         # move down
         j = i - 8
         while (j >= 0):
            if can_move(j): res.append((Player.rook, i, j))
            else:
               if self.piece_at_int(j) is None:
                  res.append((Player.rook, i, j))
               break
            j -= 8
      
      if self.knight & cell:
         if col >= 2:
            for move in [i+6, i-10]:
               if move >= 0 and move < 64 and self.piece_at_int(move) is None:
                  res.append((Player.knight, i, move))
         if col >= 1:
            for move in [i+15, i-17]:
               if move >= 0 and move < 64 and self.piece_at_int(move) is None:
                  res.append((Player.knight, i, move))
         if col <= 5:
            for move in [i+10, i-6]:
               if move >= 0 and move < 64 and self.piece_at_int(move) is None:
                  res.append((Player.knight, i, move))
         if col <= 6:
            for move in [i+17, i-15]:
               if move >= 0 and move < 64 and self.piece_at_int(move) is None:
                  res.append((Player.knight, i, move))
               
      if self.bishop & cell:
         mcol, mrow = col-1, row-1
         while mcol >= 0 and mrow >= 0:
            move = mrow * 8 + mcol
            if can_move(move): res.append((Player.bishop, i, move))
            else:
               if self.piece_at_int(move) is None:
                  res.append((Player.bishop, i, move))
               break
            mcol -= 1
            mrow -= 1

         mcol, mrow = col-1, row+1
         while mcol >= 0 and mrow < 8:
            move = mrow * 8 + mcol
            if can_move(move): res.append((Player.bishop, i, move))
            else:
               if self.piece_at_int(move) is None:
                  res.append((Player.bishop, i, move))
               break
            mcol -= 1
            mrow += 1

         mcol, mrow = col+1, row-1
         while mcol < 8 and mrow >= 0:
            move = mrow * 8 + mcol
            if can_move(move): res.append((Player.bishop, i, move))
            else:
               if self.piece_at_int(move) is None:
                  res.append((Player.bishop, i, move))
               break
            mcol += 1
            mrow -= 1

         mcol, mrow = col+1, row+1
         while mcol < 8 and mrow < 8:
            move = mrow * 8 + mcol
            if can_move(move): res.append((Player.bishop, i, move))
            else:
               if self.piece_at_int(move) is None:
                  res.append((Player.bishop, i, move))
               break
            mcol += 1
            mrow += 1
      if self.queen & cell:
         pass
      if self.king & cell:
         pass
         

     
    return res
  


class Game:
    def __init__(self):
        self.player1 = Player(white=True)
        self.player2 = Player(white=False)

    def pretty_print(self):
        board = [['.' for _ in range(8)] for _ in range(8)]
        for i in range(8):
            for j in range(8):
                piece1 = self.player1.piece_at_int(square_to_leftshift_bits(chr(ord('a') + i), j + 1))
                piece2 = self.player2.piece_at_int(square_to_leftshift_bits(chr(ord('a') + i), j + 1))
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