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
  Convert an integer corresponding to a cell to the row, col pair
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
  pieces = ['pawn', 'rook', 'knight', 'bishop', 'queen', 'king']
  pawn, rook, knight, bishop, king, queen = 'p', 'r', 'h', 'b', 'k', 'q'

  b_icons = {
     'p': '♙', 
     'r': '♖', 
     'h': '♘', 
     'b': '♗', 
     'k': '♔', 
     'q': '♕'
  }
  w_icons = {
     'p': '♟', 
     'r': '♜', 
     'h': '♞', 
     'b': '♝', 
     'k': '♚', 
     'q': '♛'
  }

  piece_values = {
        'p': 1,
        'r': 5,
        'h': 3,
        'b': 3,
        'q': 9,
        'k': 1000
   }
  
  pawn_position_values = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4],
    [0.3, 0.3, 0.3, 0.4, 0.4, 0.3, 0.3, 0.3],
    [0.2, 0.2, 0.2, 0.25, 0.25, 0.2, 0.2, 0.2],
    [0.1, 0.1, 0.1, 0.15, 0.15, 0.1, 0.1, 0.1],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0]
  ]

  knight_position_values = [
    [-0.2, -0.1, -0.05, -0.05, -0.05, -0.05, -0.1, -0.2],
    [-0.2, -0.05,  0.0,  0.0,  0.0,  0.0, -0.05, -0.2],
    [-0.1,  0.0,  0.05,  0.1,  0.1,  0.05,  0.0, -0.1],
    [-0.1,  0.02,  0.1,  0.25,  0.25,  0.1,  0.1, -0.1],
    [-0.1,  0.0,   0.1,  0.25,  0.25,  0.1,  0.0, -0.1],
    [-0.1,  0.01,  0.05,  0.3,  0.3,  0.05,  0.01, -0.1],
    [-0.2, -0.05,  0.0,  0.1,  0.1,  0.0, -0.05, -0.2],
    [-0.2, -0.1, -0.05, -0.05, -0.05, -0.05, -0.1, -0.2]
   ]
  
  king_position_values = [
    [-0.3, -0.4, -0.4, -0.5, -0.5, -0.4, -0.4, -0.3],
    [-0.3, -0.4, -0.4, -0.5, -0.5, -0.4, -0.4, -0.3],
    [-0.3, -0.4, -0.4, -0.5, -0.5, -0.4, -0.4, -0.3],
    [-0.3, -0.4, -0.4, -0.5, -0.5, -0.4, -0.4, -0.3],
    [-0.2, -0.3, -0.3, -0.4, -0.4, -0.3, -0.3, -0.2],
    [-0.1, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.1],
    [ 0.2,  0.2,  0.0,  0.0,  0.0,  0.0,  0.2,  0.2],
    [ 0.2,  0.4,  0.1,  0.0,  0.0,  0.1,  0.4,  0.2]
   ]

  def __init__(self, white: bool, copy=None):
   if copy:
      self.pawn = copy.pawn
      self.rook = copy.rook
      self.knight = copy.knight
      self.bishop = copy.bishop
      self.king = copy.king
      self.queen = copy.queen
      self.white = copy.white
      self.last_move = copy.last_move
      self.can_castle_left = copy.can_castle_left
      self.can_castle_right = copy.can_castle_right
   else:
      self.pawn = np.uint64(0)
      self.rook = np.uint64(0)
      self.knight = np.uint64(0)
      self.bishop = np.uint64(0)
      self.king = np.uint64(0)
      self.queen = np.uint64(0)
      self.reset(white)
      self.white = white
      self.last_move = None
      self.can_castle_left = True
      self.can_castle_right = True
   
  def sum_pieces(self):
      score = 0
      for piece_type in Player.pieces:
         pieces = getattr(self, piece_type)
         score += bin(pieces).count('1') * Player.piece_values[getattr(Player, piece_type)]
      return score
  
  def sum_piece_bonuses(self): 
      score = 0
      for piece_type in Player.pieces:
         pieces = getattr(self, piece_type)
         score += bin(pieces).count('1') * Player.piece_values[getattr(Player, piece_type)]
         if piece_type == 'pawn' or piece_type == 'knight' or piece_type == 'king':
            for i in range(8):
               for j in range(8):
                  if piece_type == 'pawn' and pieces & np.uint64(1 << (i * 8 + j)):
                     if self.white:
                        score += Player.pawn_position_values[7-i][j]
                     else:
                        score += Player.pawn_position_values[i][j]
                  elif piece_type == 'knight' and pieces & np.uint64(1 << (i * 8 + j)):
                     if self.white:
                        score += Player.pawn_position_values[7-i][j]
                     else:
                        score += Player.pawn_position_values[i][j]
                  elif piece_type == 'king' and pieces & np.uint64(1 << (i * 8 + j)):
                     if self.white:
                        score += Player.pawn_position_values[7-i][j]
                     else:
                        score += Player.pawn_position_values[i][j]
      return score
  
  def naive_score(self, other):
      if self.white: return self.sum_pieces() - other.sum_pieces()
      return other.sum_pieces() - self.sum_pieces()
  
  def positional_score(self, other):
     if self.white: return self.sum_piece_bonuses() - other.sum_piece_bonuses()
     return other.sum_piece_bonuses() - self.sum_piece_bonuses()

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

  def remove_piece_at_int(self, num: int) -> None:
      if num < 0 or num >= 64: return None
      num = np.uint64(1 << num)
      if self.pawn & num: self.pawn = self.pawn ^ num
      if self.rook & num: self.rook = self.rook ^ num
      if self.knight & num: self.knight = self.knight ^ num
      if self.bishop & num: self.bishop = self.bishop ^ num
      if self.queen & num: self.queen = self.queen ^ num
      if self.king & num: self.king = self.king ^ num


  def is_square_under_attack(self, square: int, other):
        def contains_piece(square):
           return self.piece_at_int(square) is not None or other.piece_at_int(square) is not None
           
        for i in range(64):
            cell = np.uint64(1 << i)
            if other.piece_at_int(i):
                col = i % 8
                row = int(i / 8)

                if other.piece_at_int(i) == Player.pawn:
                    sign = -1 if self.white else 1
                    attack_left = i + 7 * sign
                    attack_right = i + 9 * sign
                    if square in [attack_left, attack_right] and abs((square % 8) - col) == 1:
                        return True
                
                if other.piece_at_int(i) == Player.rook or other.piece_at_int(i) == Player.queen:
                    if square % 8 == col or square // 8 == row:
                        if row == square // 8: 
                           min_col = min(col, square % 8)
                           max_col = max(col, square % 8)
                           for c in range(min_col + 1, max_col):
                              if contains_piece(row*8 + c):
                                 break
                           else:
                              return True
                        
                        if square % 8 == col:
                              min_row = min(row, square // 8) + 1
                              max_row = max(row, square // 8)
                              for j in range(min_row, max_row):
                                 if contains_piece(j * 8 + col):
                                    break
                              else:
                                 return True
                
                if other.piece_at_int(i) == Player.knight:
                     knight_moves = [-17, -15, -10, -6, 6, 10, 15, 17]
                     for move in knight_moves:
                        if 0 <= i + move < 64:
                              if abs(col - (i + move) % 8) in [1, 2]:
                                 if i + move == square:
                                    return True
                
                if other.piece_at_int(i) == Player.bishop or other.piece_at_int(i) == Player.queen:
                    if abs((col) - (square % 8)) == abs((row) - (square // 8)):
                        min_col = min(col, square % 8) + 1
                        max_col = max(col, square % 8)
                        min_row = min(row, square // 8) + 1
                        max_row = max(row, square // 8)
                        for c, r in zip(range(min_col, max_col), range(min_row, max_row)):
                              if contains_piece(r * 8 + c):
                                 break
                        else:
                              return True
                
                if other.piece_at_int(i) == Player.king:
                     king_moves = [-9, -8, -7, -1, 1, 7, 8, 9]
                     for move in king_moves:
                        if 0 <= i + move < 64:
                              if abs(col - (i + move) % 8) <= 1 and abs(row - (i + move) // 8) <= 1:
                                 if i + move == square:
                                    return True
        
        return False 

  def is_king_under_attack(self, other):
     sq = None
     for i in range(64):
        if self.king & np.uint64(1 << i):
           sq = i
     return self.is_square_under_attack(sq, other)

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
        for pos in [i + 7 * sign, i + 9 * sign]:
            if abs(pos % 8 - col) == 1:
               if other.piece_at_int(pos): 
                  if (self.white and int(pos / 8) == 7) or (not self.white and int(pos / 8) == 0):
                     for piece in [Player.bishop, Player.knight, Player.rook, Player.queen]:
                        res.append((Player.pawn, i, pos, piece))
                  else:
                     res.append((Player.pawn, i, pos))

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

      if self.rook & cell or self.queen & cell:
         piece = Player.rook if self.rook & cell else Player.queen
         # move left
         j = i - 1
         while j >= 0:
            if can_move(j): res.append((piece, i, j))
            else:
               if self.piece_at_int(j) is None:
                  res.append((piece, i, j))
               break
            if j % 8 == 0: break
            j -= 1

         # move right
         j = i + 1
         while j < 64 and j % 8 != 0:
            if can_move(j): res.append((piece, i, j))
            else:
               if self.piece_at_int(j) is None:
                  res.append((piece, i, j))
               break
            j += 1
         
         # move up
         j = i + 8
         while (j < 64):
            if can_move(j): res.append((piece, i, j))
            else:
               if self.piece_at_int(j) is None:
                  res.append((piece, i, j))
               break
            j += 8
         
         # move down
         j = i - 8
         while (j >= 0):
            if can_move(j): res.append((piece, i, j))
            else:
               if self.piece_at_int(j) is None:
                  res.append((piece, i, j))
               break
            j -= 8
      
      if self.knight & cell:
         knight_moves = [-17, -15, -10, -6, 6, 10, 15, 17]
         for move in knight_moves:
            if 0 <= i + move < 64 and self.piece_at_int(i+move) is None:
                  if abs(i % 8 - (i + move) % 8) in [1, 2]:
                     res.append((Player.knight, i, i+move))
               
      if self.bishop & cell or self.queen & cell:
         piece = Player.bishop if self.bishop & cell else Player.queen
         mcol, mrow = col-1, row-1
         while mcol >= 0 and mrow >= 0:
            move = mrow * 8 + mcol
            if can_move(move): res.append((piece, i, move))
            else:
               if self.piece_at_int(move) is None:
                  res.append((piece, i, move))
               break
            mcol -= 1
            mrow -= 1

         mcol, mrow = col-1, row+1
         while mcol >= 0 and mrow < 8:
            move = mrow * 8 + mcol
            if can_move(move): res.append((piece, i, move))
            else:
               if self.piece_at_int(move) is None:
                  res.append((piece, i, move))
               break
            mcol -= 1
            mrow += 1

         mcol, mrow = col+1, row-1
         while mcol < 8 and mrow >= 0:
            move = mrow * 8 + mcol
            if can_move(move): res.append((piece, i, move))
            else:
               if self.piece_at_int(move) is None:
                  res.append((piece, i, move))
               break
            mcol += 1
            mrow -= 1

         mcol, mrow = col+1, row+1
         while mcol < 8 and mrow < 8:
            move = mrow * 8 + mcol
            if can_move(move): res.append((piece, i, move))
            else:
               if self.piece_at_int(move) is None:
                  res.append((piece, i, move))
               break
            mcol += 1
            mrow += 1

      if self.king & cell:
         for x_offset in [-1, 0, 1]:
            for y_offset in [-1, 0, 1]:
               if x_offset == 0 and y_offset == 0:
                  continue
               
               new_col = col + x_offset
               new_row = row + y_offset
               new_cell = new_row * 8 + new_col
               
               if new_col < 0 or new_col >= 8 or new_row < 0 or new_row >= 8:
                  continue 
                  
               if can_move(new_cell) or other.piece_at_int(new_cell):
                  res.append((Player.king, i, new_cell))
         
         if row == 7 or row == 0:
            if self.can_castle_left:
               if can_move(i-1) and can_move(i-2) and can_move(i-3):
                  if not self.is_square_under_attack(i, other) and \
                     not self.is_square_under_attack(i-1, other) and \
                     not self.is_square_under_attack(i-2, other):
                     res.append((Player.king, i, i-2))
            if self.can_castle_right:
               if can_move(i+1) and can_move(i+2):
                  if not self.is_square_under_attack(i, other) and \
                     not self.is_square_under_attack(i+1, other) and \
                     not self.is_square_under_attack(i+2, other):
                     res.append((Player.king, i, i+2))
     
    return res
  
  def make_move(self, move, other):
      copy = Player(self.white, self)
      other_copy = Player(other.white, other)
      
      if move[0] == Player.pawn:
         copy.pawn = copy.pawn ^ np.uint64(1 << move[1])
         # promoting a pawn
         if len(move) == 4:
            if move[3] == Player.bishop: copy.bishop = copy.bishop | np.uint64(1 << move[2])
            elif move[3] == Player.rook: copy.rook = copy.rook | np.uint64(1 << move[2])
            elif move[3] == Player.knight: copy.knight = copy.knight | np.uint64(1 << move[2])
            elif move[3] == Player.queen: copy.queen = copy.queen | np.uint64(1 << move[2])
         else:
            copy.pawn = copy.pawn | np.uint64(1 << move[2])
         
         # en passant
         if ((move[1] % 8) - (move[2] % 8)) == 1 and other_copy.piece_at_int(move[2]) is None:
            other_copy.remove_piece_at_int(move[1]-1)
         elif ((move[2] % 8) - (move[1] % 8)) == 1 and other_copy.piece_at_int(move[2]) is None:
            other_copy.remove_piece_at_int(move[1]+1)

      elif move[0] == Player.rook:
         copy.rook = (copy.rook ^ np.uint64(1 << move[1])) | np.uint64(1 << move[2])
      elif move[0] == Player.knight:
         copy.knight = (copy.knight ^ np.uint64(1 << move[1])) | np.uint64(1 << move[2])
      elif move[0] == Player.bishop:
         copy.bishop = (copy.bishop ^ np.uint64(1 << move[1])) | np.uint64(1 << move[2])
      elif move[0] == Player.queen:
         copy.queen = (copy.queen ^ np.uint64(1 << move[1])) | np.uint64(1 << move[2])
      elif move[0] == Player.king:
         copy.king = copy.king ^ np.uint64(1 << move[1])
         # castling left
         if (move[1] % 8) - (move[2] % 8) == 2:
            if copy.white:
               copy.rook = (copy.rook ^ np.uint64(1)) | np.uint64(1 << 3)
            else:
               copy.rook = (copy.rook ^ np.uint64(1 << 54)) | np.uint64(1 << 57)
         # castling right
         elif (move[2] % 8) - (move[2] % 8) == 2:
            if copy.white:
               copy.rook = (copy.rook ^ np.uint64(1 << 7)) | np.uint64(1 << 5)
            else:
               copy.rook = (copy.rook ^ np.uint64(1 << 63)) | np.uint64(1 << 61)
         copy.king = copy.king | np.uint64(1 << move[2])

      other_copy.remove_piece_at_int(move[2])

      copy.last_move = move
      if move[0] == Player.king:
         copy.can_castle_left = False
         copy.can_castle_right = False
      elif move[0] == Player.rook:
         if move[1] % 8 == 0:
            copy.can_castle_left = False
         elif move[1] % 8 == 7:
            copy.can_castle_right = False
            
      return copy, other_copy
     
  


class Game:
    def __init__(self, player_white=True, hard=False):
        self.cpu = Player(white=not player_white)
        self.player = Player(white=player_white)
        self.player_white = player_white
        self.hard = hard

    def player_move(self, move):
         if move in self.player.possible_moves(self.cpu):
            self.player, self.cpu = self.player.make_move(move, self.cpu)
         else:
            raise Exception("Invalid Move")
    
    def cpu_move(self, move):
         self.cpu, self.player = self.cpu.make_move(move, self.player)

    def minimax(self, player, other, depth=6):
       depth = min(depth, 6)
       best_move = None
       best_move_val = float('-inf') if player.white else float('inf')
       for move in player.possible_moves(other):
          player_moved, other_moved = player.make_move(move, other)
          if player_moved.is_king_under_attack(other_moved): continue
          if depth > 0:
             _, val = self.minimax(other_moved, player_moved, depth-1)
          else:
             if self.hard:
                val = player_moved.positional_score(other_moved)
             else:
                val = player_moved.naive_score(other_moved)

          if player.white:
               if val > best_move_val:
                  best_move = move
                  best_move_val = val
          else:
               if val < best_move_val:
                  best_move = move
                  best_move_val = val

       return best_move, best_move_val


    def pretty_print(self):
        board = [['.' for _ in range(8)] for _ in range(8)]
        for i in range(8):
            for j in range(8):
                cpu_piece = self.cpu.piece_at_int(square_to_leftshift_bits(chr(ord('a') + j), i + 1))
                player_piece = self.player.piece_at_int(square_to_leftshift_bits(chr(ord('a') + j), i + 1))
                if cpu_piece:
                    board[i][j] = ('b' if self.player_white else 'w') + cpu_piece
                elif player_piece:
                    board[i][j] = ('w' if self.player_white else 'b') + player_piece

        print("    a   b   c   d   e   f   g   h")
        print("    -----------------------------")
        for i in range(7, -1, -1):
            print(f" {i+1} ", end="")
            for j in range(8):
                piece = board[i][j]
                if piece[0] == 'w':
                    print(f" {Player.w_icons[piece[1]]} ", end=" ")  # White piece
                elif piece[0] == 'b':
                    print(f" {Player.b_icons[piece[1]]} ", end=" ")  # Black piece
                else:
                    print(" . ", end=" ")  # Empty square
            print(f" {i+1}")
        print("    -----------------------------")
        print("    a   b   c   d   e   f   g   h")