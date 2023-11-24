import numpy as np
from typing import Optional, Tuple

'''
square: chess board label for a cell (e.g. 'c3')
int: 64-bit integer corresponding to board state for a piece
'''

def square_to_int(row: str, col: int) -> Optional[int]:
  '''
  Convert a row, col pair into integer representation
  '''
  if len(row) == 1 and row >= 'a' and row <= 'h' and col >= 1 and col <= 8:
    c = 1 << ((col - 1) * 8 + (ord(row) - ord('a')))
    return np.uint64(c)

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
  def __init__(self, color):
    pass
