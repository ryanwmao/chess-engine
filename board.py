from typing import Optional, Tuple, List

"""
square: chess board label for a cell (e.g. 'c3')
int: 64-bit integer corresponding to board state for a piece
"""


king_lookup_table = {(1 << i): i for i in range(64)}
COL_A = 0x0101010101010101
COL_B = 0x0202020202020202
COL_C = 0x0404040404040404
COL_D = 0x0808080808080808
COL_E = 0x1010101010101010
COL_F = 0x2020202020202020
COL_G = 0x4040404040404040
COL_H = 0x8080808080808080

ROW_1 = 0x00000000000000FF
ROW_2 = 0x000000000000FF00
ROW_3 = 0x0000000000FF0000
ROW_4 = 0x00000000FF000000
ROW_5 = 0x000000FF00000000
ROW_6 = 0x0000FF0000000000
ROW_7 = 0x00FF000000000000
ROW_8 = 0xFF00000000000000

def position_row(position):
    for i, row in enumerate([ROW_1, ROW_2, ROW_3, ROW_4, ROW_5, ROW_6, ROW_7, ROW_8]):
        if position & row:
            return i, row

def position_col(position):
    for i, col in enumerate([COL_A, COL_B, COL_C, COL_D, COL_E, COL_F, COL_G, COL_H]):
        if position & col:
            return i, col

def square_to_int(row: str, col: int) -> Optional[int]:
    """
    Convert a row, col pair into integer representation
    """
    if len(row) == 1 and row >= "a" and row <= "h" and col >= 1 and col <= 8:
        return (1 << ((col - 1) * 8 + (ord(row) - ord("a"))))


def square_to_leftshift_bits(row: str, col: int) -> Optional[int]:
    """
    Convert a row, col pair into the number of left shift bits
    """
    if len(row) == 1 and row >= "a" and row <= "h" and col >= 1 and col <= 8:
        return (col - 1) * 8 + (ord(row) - ord("a"))


def int_to_square(num: int) -> Optional[Tuple[str, int]]:
    """
    Convert an integer corresponding to a cell to the row, col pair
    """
    if bin(num).count("1") == 1:
        lshift = king_lookup_table[num]
        col = int(lshift / 8) + 1
        row = chr(lshift % 8 + ord("a"))
        return row, col


class Player:
    pieces = ["pawn", "rook", "knight", "bishop", "queen", "king"]
    pawn, rook, knight, bishop, king, queen = "p", "r", "h", "b", "k", "q"

    b_icons = {"p": "♙", "r": "♖", "h": "♘", "b": "♗", "k": "♔", "q": "♕"}
    w_icons = {"p": "♟", "r": "♜", "h": "♞", "b": "♝", "k": "♚", "q": "♛"}

    piece_values = {"p": 1, "r": 5, "h": 3, "b": 3, "q": 9, "k": 1000}


    pawn_0_5_values = 0b11111111 << 48
    pawn_0_4_values = 0b1111111100011 << 35
    pawn_0_3_values = 0b11100111 << 32
    pawn_0_25_values = 0b11 << 27
    pawn_0_16_values = 0b11100111 << 24
    pawn_0_15_values = 0b11 << 19
    pawn_0_1_values = 0b11100111 << 16
    pawn_position_values = [(0.5, pawn_0_5_values), (0.4, pawn_0_4_values), (0.3, pawn_0_3_values), (0.25, pawn_0_25_values), (0.16, pawn_0_16_values), (0.15, pawn_0_15_values), (0.1, pawn_0_1_values)]

    knight_n0_2_values  = 0b1000000110000001000000000000000000000000000000001000000110000001
    knight_n0_1_values  = 0b0100001000000000100000011000000110000001100000010000000001000010
    knight_n0_05_values = 0b0011110001000010000000000000000000000000000000000100001000111100
    knight_0_05_values  = 0b0000000000000000001001000000000000000000001001000000000000000000
    knight_0_1_values   = 0b0000000000000000000110000110011000100100000000000001100000000000
    knight_0_25_values  = 0b0000000000000000000000000001100000011000000000000000000000000000
    knight_0_3_values   = 0b11 << 19
    knight_position_values = [(-0.2, knight_n0_2_values), (-0.1, knight_n0_1_values), (-0.05, knight_n0_05_values), (0.05, knight_0_05_values), (0.1, knight_0_1_values), (0.25, knight_0_25_values), (0.3, knight_0_3_values)]

    king_n0_1_values = 0b10000001 << 16
    king_n0_2_values = 0b10000001011111100000000000000000
    king_n0_3_values = 0b1000000110000001100000011000000101100110000000000000000000000000
    king_n0_4_values = 0b0110011001100110011001100110011000011000000000000000000000000000
    king_n0_5_values = 0b0001100000011000000110000001100000000000000000000000000000000000
    king_0_1_values  = 0b100100
    king_0_2_values  = 0b1100001110000001
    king_0_4_values  = 0b1000010
    king_position_values = [(-0.1, king_n0_1_values), (-0.2, king_n0_2_values), (-0.3, king_n0_3_values), (-0.4, king_n0_4_values), (-0.5, king_n0_5_values), (0.1, king_0_1_values), (0.2, king_0_2_values), (0.4, king_0_4_values)]

    # pawn_position_values = [
    #     [0, 0, 0, 0, 0, 0, 0, 0],
    #     [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    #     [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4],
    #     [0.3, 0.3, 0.3, 0.4, 0.4, 0.3, 0.3, 0.3],
    #     [0.2, 0.2, 0.2, 0.25, 0.25, 0.2, 0.2, 0.2],
    #     [0.1, 0.1, 0.1, 0.15, 0.15, 0.1, 0.1, 0.1],
    #     [0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0],
    # ]

    # knight_position_values = [
    #     [-0.2, -0.1, -0.05, -0.05, -0.05, -0.05, -0.1, -0.2],
    #     [-0.2, -0.05, 0.0, 0.0, 0.0, 0.0, -0.05, -0.2],
    #     [-0.1, 0.0, 0.05, 0.1, 0.1, 0.05, 0.0, -0.1],
    #     [-0.1, 0.1, 0.1, 0.25, 0.25, 0.1, 0.1, -0.1],
    #     [-0.1, 0.0, 0.1, 0.25, 0.25, 0.1, 0.0, -0.1],
    #     [-0.1, 0.01, 0.05, 0.3, 0.3, 0.05, 0.01, -0.1],
    #     [-0.2, -0.05, 0.0, 0.1, 0.1, 0.0, -0.05, -0.2],
    #     [-0.2, -0.1, -0.05, -0.05, -0.05, -0.05, -0.1, -0.2],
    # ]

    # king_position_values = [
    #     [-0.3, -0.4, -0.4, -0.5, -0.5, -0.4, -0.4, -0.3],
    #     [-0.3, -0.4, -0.4, -0.5, -0.5, -0.4, -0.4, -0.3],
    #     [-0.3, -0.4, -0.4, -0.5, -0.5, -0.4, -0.4, -0.3],
    #     [-0.3, -0.4, -0.4, -0.5, -0.5, -0.4, -0.4, -0.3],
    #     [-0.2, -0.3, -0.3, -0.4, -0.4, -0.3, -0.3, -0.2],
    #     [-0.1, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.1],
    #     [0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.2, 0.2],
    #     [0.2, 0.4, 0.1, 0.0, 0.0, 0.1, 0.4, 0.2],
    # ]

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
            self.pawn = 0
            self.rook = 0
            self.knight = 0
            self.bishop = 0
            self.king = 0
            self.queen = 0
            self.reset(white)
            self.white = white
            self.last_move = None
            self.can_castle_left = True
            self.can_castle_right = True

    def sum_pieces(self):
        score = 0
        for piece_type in Player.pieces:
            pieces = getattr(self, piece_type)
            score += (
                bin(pieces).count("1")
                * Player.piece_values[getattr(Player, piece_type)]
            )
        return score

    def sum_piece_bonuses(self):
        score = self.sum_pieces()
        for bonus, mask in Player.pawn_position_values:
            if self.white:
                score += bin(self.pawn & mask).count("1") * bonus
            else:
                score += bin(int('{:064b}'.format(self.pawn)[::-1], 2) & mask).count("1") * bonus
        for bonus, mask in Player.knight_position_values:
            if self.white:
                score += bin(self.knight & mask).count("1") * bonus
            else:
                score += bin(int('{:064b}'.format(self.knight)[::-1], 2) & mask).count("1") * bonus
        for bonus, mask in Player.king_position_values:
            if self.white:
                score += bin(self.king & mask).count("1") * bonus
            else:
                score += bin(int('{:064b}'.format(self.king)[::-1], 2) & mask).count("1") * bonus
        return score

    def naive_score(self, other):
        if self.white:
            return self.sum_pieces() - other.sum_pieces()
        return other.sum_pieces() - self.sum_pieces()

    def positional_score(self, other):
        if self.white:
            return self.sum_piece_bonuses() - other.sum_piece_bonuses()
        return other.sum_piece_bonuses() - self.sum_piece_bonuses()

    def reset(self, white: bool) -> None:
        self.pawn = 0
        if white:
            for i in range(8):
                self.pawn = self.pawn | (1 << (8 + i))
            self.rook = (1) | (1 << 7)
            self.knight = (2) | (1 << 6)
            self.bishop = (4) | (1 << 5)
            self.queen = (8)
            self.king = (16)
        else:
            for i in range(8):
                self.pawn = self.pawn | (1 << (48 + i))
            self.rook = (1 << 56) | (1 << 63)
            self.knight = (1 << 57) | (1 << 62)
            self.bishop = (1 << 58) | (1 << 61)
            self.queen = (1 << 59)
            self.king = (1 << 60)

    def piece_at_int(self, num: int) -> Optional[str]:
        """
        Returns a piece at a cell, if one exists.
        Input is the number of left shift bits.
        """
        if num < 0 or num >= 64:
            return None
        num = (1 << num)
        if self.pawn & num:
            return Player.pawn
        elif self.rook & num:
            return Player.rook
        elif self.knight & num:
            return Player.knight
        elif self.bishop & num:
            return Player.bishop
        elif self.queen & num:
            return Player.queen
        elif self.king & num:
            return Player.king
    
    def union_pieces(self):
        return self.pawn | self.rook | self.knight | self.bishop | self.queen | self.king
    
    def piece_exists(self, num):
        if num < 0 or num >= 64:
            return False
        return True if (self.union_pieces() & (1 << num)) else False

    def remove_piece_at_int(self, num: int) -> None:
        if num < 0 or num >= 64:
            return None
        num = (1 << num)
        if self.pawn & num:
            self.pawn = self.pawn ^ num
        elif self.rook & num:
            self.rook = self.rook ^ num
        elif self.knight & num:
            self.knight = self.knight ^ num
        elif self.bishop & num:
            self.bishop = self.bishop ^ num
        elif self.queen & num:
            self.queen = self.queen ^ num
        elif self.king & num:
            self.king = self.king ^ num

    def get_pawn_attacks(self, position):
        if self.white:
            return ((position << (7)) & ~COL_H) | ((position << (9)) & ~COL_A)
        else:
            return ((position >> (7)) & ~COL_A) | ((position >> (9)) & ~COL_H)

    def get_knight_attacks(position):
        l1 = (position >> (1)) & ~COL_H
        l2 = (position >> (2)) & ~(COL_G | COL_H)
        r1 = (position << (1)) & ~COL_A
        r2 = (position << (2)) & ~(COL_A | COL_B)
        h1 = l1 | r1
        h2 = l2 | r2
        return (h1 << (16)) | (h1 >> (16)) | (h2 << (8)) | (h2 >> (8))

    def get_king_attacks(position):
        attacks = ((position << (1)) & ~COL_H) | \
                  ((position >> (1)) & ~COL_A) | \
                  ((position << (8)) & ~ROW_1)  | \
                  ((position >> (8)) & ~ROW_8)  | \
                  ((position << (7)) & ~(ROW_1 | COL_A)) | \
                  ((position >> (7)) & ~(ROW_8 | COL_H)) | \
                  ((position << (9)) & ~(ROW_1 | COL_H)) | \
                  ((position >> (9)) & ~(ROW_8 | COL_A)) 
        return attacks

    # does NOT check for en passant!!
    def is_square_under_attack(self, square: int, other):
        def contains_piece(square):
            return (
                self.piece_exists(square) or other.piece_exists(square)
            )

        position = (1 << square)
        if other.pawn & self.get_pawn_attacks(position): return True
        elif other.knight & Player.get_knight_attacks(position): return True
        elif other.king & Player.get_king_attacks(position): return True
        else:
            row, _ = position_row(position)
            col, _ = position_col(position)
            for c in range(col+1, 8):
                i = row * 8 + c
                if self.piece_exists(i):
                    break
                if other.piece_exists(i):
                    p = other.piece_at_int(i)
                    if p == 'r' or p == 'q':
                        return True
                    else:
                        break
            for c in range(col-1, -1, -1):
                i = row * 8 + c
                if self.piece_exists(i):
                    break
                if other.piece_exists(i):
                    p = other.piece_at_int(i)
                    if p == 'r' or p == 'q':
                        return True
                    else:
                        break
            for r in range(row+1, 8):
                i = r * 8 + col
                if self.piece_exists(i):
                    break
                if other.piece_exists(i):
                    p = other.piece_at_int(i)
                    if p == 'r' or p == 'q':
                        return True
                    else:
                        break
            for r in range(row-1, -1, -1):
                i = r * 8 + col
                if self.piece_exists(i):
                    break
                if other.piece_exists(i):
                    p = other.piece_at_int(i)
                    if p == 'r' or p == 'q':
                        return True
                    else:
                        break
            
            r, c = row, col
            while r > -1 and c > -1:
                r -= 1
                c -= 1
                i = r * 8 + c
                if self.piece_exists(i):
                    break
                if other.piece_exists(i):
                    p = other.piece_at_int(i)
                    if p == 'b' or p == 'q':
                        return True
                    else:
                        break
            r, c = row, col
            while r > -1 and c < 8:
                r -= 1
                c += 1
                i = r * 8 + c
                if self.piece_exists(i):
                    break
                if other.piece_exists(i):
                    p = other.piece_at_int(i)
                    if p == 'b' or p == 'q':
                        return True
                    else:
                        break
            
            r, c = row, col
            while r < 8 and c > -1:
                r += 1
                c -= 1
                i = r * 8 + c
                if self.piece_exists(i):
                    break
                if other.piece_exists(i):
                    p = other.piece_at_int(i)
                    if p == 'b' or p == 'q':
                        return True
                    else:
                        break
            
            r, c = row, col
            while r < 8 and c < 8:
                r += 1
                c += 1
                i = r * 8 + c
                if self.piece_exists(i):
                    break
                if other.piece_exists(i):
                    p = other.piece_at_int(i)
                    if p == 'b' or p == 'q':
                        return True
                    else:
                        break

    def is_king_under_attack(self, other):
        sq = king_lookup_table[self.king]
        return self.is_square_under_attack(sq, other)

    # we don't account for pins under the assumption that exposing the king
    # by moving the pin will be avoided through our reward function
    def possible_moves(self, other):
        def can_move(square):
            return (
                not other.piece_exists(square) and not self.piece_exists(square)
            )

        res = []
        for i in range(0, 64):
            cell = (1 << i)
            col = i % 8
            row = int(i / 8)
            if self.pawn & cell:
                sign = 1 if self.white else -1
                # pawn taking diagonally
                for pos in [i + 7 * sign, i + 9 * sign]:
                    if abs(pos % 8 - col) == 1:
                        if other.piece_exists(pos):
                            if (self.white and int(pos / 8) == 7) or (
                                not self.white and int(pos / 8) == 0
                            ):
                                for piece in [
                                    Player.bishop,
                                    Player.knight,
                                    Player.rook,
                                    Player.queen,
                                ]:
                                    res.append((Player.pawn, i, pos, piece))
                            else:
                                res.append((Player.pawn, i, pos))

                # pawn pushing forward, or pushing double from start
                move_forward, move_double = i + 8 * sign, i + 16 * sign
                if move_forward < 64 and move_forward >= 0:
                    if can_move(move_forward):
                        # promote pawn
                        if (self.white and int(move_forward / 8) == 7) or (
                            not self.white and int(move_forward / 8) == 0
                        ):
                            for piece in [
                                Player.bishop,
                                Player.knight,
                                Player.rook,
                                Player.queen,
                            ]:
                                res.append((Player.pawn, i, move_forward, piece))
                        else:
                            res.append((Player.pawn, i, move_forward))
                        # push two spaces from start
                        if (
                            (int(i / 8) == 1 and self.white)
                            or (int(i / 8) == 6)
                            and not self.white
                        ) and can_move(move_double):
                            res.append((Player.pawn, i, move_double))

                # en passant
                if (
                    other.last_move
                    and other.last_move[0] == Player.pawn
                    and abs(other.last_move[1] - other.last_move[2]) == 16
                ):
                    if other.last_move[2] == i - 1 * sign:
                        res.append((Player.pawn, i, i + 7 * sign))
                    if other.last_move[2] == i + 1 * sign:
                        res.append((Player.pawn, i, i + 9 * sign))

            if self.rook & cell or self.queen & cell:
                piece = Player.rook if self.rook & cell else Player.queen
                # move left
                j = i - 1
                while j >= (i // 8) * 8:
                    if can_move(j):
                        res.append((piece, i, j))
                    else:
                        if not self.piece_exists(j):
                            res.append((piece, i, j))
                        break
                    if j % 8 == 0:
                        break
                    j -= 1

                # move right
                j = i + 1
                while j < 64 and j % 8 != 0:
                    if can_move(j):
                        res.append((piece, i, j))
                    else:
                        if not self.piece_exists(j):
                            res.append((piece, i, j))
                        break
                    j += 1

                # move up
                j = i + 8
                while j < 64:
                    if can_move(j):
                        res.append((piece, i, j))
                    else:
                        if not self.piece_exists(j):
                            res.append((piece, i, j))
                        break
                    j += 8

                # move down
                j = i - 8
                while j >= 0:
                    if can_move(j):
                        res.append((piece, i, j))
                    else:
                        if not self.piece_exists(j):
                            res.append((piece, i, j))
                        break
                    j -= 8

            if self.knight & cell:
                knight_moves = [-17, -15, -10, -6, 6, 10, 15, 17]
                for move in knight_moves:
                    if 0 <= i + move < 64 and not self.piece_exists(i + move):
                        if abs(i % 8 - (i + move) % 8) in [1, 2]:
                            res.append((Player.knight, i, i + move))

            if self.bishop & cell or self.queen & cell:
                piece = Player.bishop if self.bishop & cell else Player.queen
                mcol, mrow = col - 1, row - 1
                while mcol >= 0 and mrow >= 0:
                    move = mrow * 8 + mcol
                    if can_move(move):
                        res.append((piece, i, move))
                    else:
                        if not self.piece_exists(move):
                            res.append((piece, i, move))
                        break
                    mcol -= 1
                    mrow -= 1

                mcol, mrow = col - 1, row + 1
                while mcol >= 0 and mrow < 8:
                    move = mrow * 8 + mcol
                    if can_move(move):
                        res.append((piece, i, move))
                    else:
                        if not self.piece_exists(move):
                            res.append((piece, i, move))
                        break
                    mcol -= 1
                    mrow += 1

                mcol, mrow = col + 1, row - 1
                while mcol < 8 and mrow >= 0:
                    move = mrow * 8 + mcol
                    if can_move(move):
                        res.append((piece, i, move))
                    else:
                        if not self.piece_exists(move):
                            res.append((piece, i, move))
                        break
                    mcol += 1
                    mrow -= 1

                mcol, mrow = col + 1, row + 1
                while mcol < 8 and mrow < 8:
                    move = mrow * 8 + mcol
                    if can_move(move):
                        res.append((piece, i, move))
                    else:
                        if not self.piece_exists(move):
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

                        if can_move(new_cell) or other.piece_exists(new_cell):
                            res.append((Player.king, i, new_cell))

                if row == 7 or row == 0:
                    if self.can_castle_left:
                        if can_move(i - 1) and can_move(i - 2) and can_move(i - 3):
                            if (
                                not self.is_square_under_attack(i, other)
                                and not self.is_square_under_attack(i - 1, other)
                                and not self.is_square_under_attack(i - 2, other)
                            ):
                                res.append((Player.king, i, i - 2))
                    if self.can_castle_right:
                        if can_move(i + 1) and can_move(i + 2):
                            if (
                                not self.is_square_under_attack(i, other)
                                and not self.is_square_under_attack(i + 1, other)
                                and not self.is_square_under_attack(i + 2, other)
                            ):
                                res.append((Player.king, i, i + 2))

        return res

    def make_move(self, move, other):
        copy = Player(self.white, self)
        other_copy = Player(other.white, other)

        if move[0] == Player.pawn:
            copy.pawn = copy.pawn ^ (1 << move[1])
            # promoting a pawn
            if len(move) == 4:
                if move[3] == Player.bishop:
                    copy.bishop = copy.bishop | (1 << move[2])
                elif move[3] == Player.rook:
                    copy.rook = copy.rook | (1 << move[2])
                elif move[3] == Player.knight:
                    copy.knight = copy.knight | (1 << move[2])
                elif move[3] == Player.queen:
                    copy.queen = copy.queen | (1 << move[2])
            else:
                copy.pawn = copy.pawn | (1 << move[2])

            # en passant
            if ((move[1] % 8) - (move[2] % 8)) == 1 and not other_copy.piece_exists(
                move[2]
            ):
                other_copy.remove_piece_at_int(move[1] - 1)
            elif ((move[2] % 8) - (move[1] % 8)) == 1 and not other_copy.piece_exists(
                move[2]
            ):
                other_copy.remove_piece_at_int(move[1] + 1)

        elif move[0] == Player.rook:
            copy.rook = (copy.rook ^ (1 << move[1])) | (1 << move[2])
        elif move[0] == Player.knight:
            copy.knight = (copy.knight ^ (1 << move[1])) | (
                1 << move[2]
            )
        elif move[0] == Player.bishop:
            copy.bishop = (copy.bishop ^ (1 << move[1])) | (
                1 << move[2]
            )
        elif move[0] == Player.queen:
            copy.queen = (copy.queen ^ (1 << move[1])) | (
                1 << move[2]
            )
        elif move[0] == Player.king:
            copy.king = copy.king ^ (1 << move[1])
            # castling left
            if (move[1] % 8) - (move[2] % 8) == 2:
                if copy.white:
                    copy.rook = (copy.rook ^ (1)) | (1 << 3)
                else:
                    copy.rook = (copy.rook ^ (1 << 56)) | (1 << 59)
            # castling right
            elif (move[2] % 8) - (move[1] % 8) == 2:
                if copy.white:
                    copy.rook = (copy.rook ^ (1 << 7)) | (1 << 5)
                else:
                    copy.rook = (copy.rook ^ (1 << 63)) | (1 << 61)
            copy.king = copy.king | (1 << move[2])

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
        best_move = None
        best_move_val = float("-inf") if player.white else float("inf")
        for move in player.possible_moves(other):
            player_moved, other_moved = player.make_move(move, other)
            if player_moved.king == 0:
                continue
            if player_moved.is_king_under_attack(other_moved):
                continue
            if depth > 0:
                _, val = self.minimax(other_moved, player_moved, depth - 1)
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

    def minimax_ab(
        self, player, other, depth=6, alpha=float("-inf"), beta=float("inf")
    ):
        best_move = None

        best_move_val = float("-inf") if player.white else float("inf")
        for move in player.possible_moves(other):
            player_moved, other_moved = player.make_move(move, other)
            if player_moved.king == 0:
                continue
            if player_moved.is_king_under_attack(other_moved):
                continue
            if depth > 0:
                _, val = self.minimax_ab(other_moved, player_moved, depth - 1, alpha, beta)
            else:
                if self.hard:
                    val = player_moved.positional_score(other_moved)
                else:
                    val = player_moved.naive_score(other_moved)

            if player.white:
                if val > best_move_val:
                    best_move = move
                    best_move_val = val
                alpha = max(alpha, val)
            else:
                if val < best_move_val:
                    best_move = move
                    best_move_val = val
                beta = min(beta, val)
            if alpha >= beta:
                break
        return best_move, best_move_val

    def pretty_print(self):
        board = [["." for _ in range(8)] for _ in range(8)]
        for i in range(8):
            for j in range(8):
                cpu_piece = self.cpu.piece_at_int(
                    square_to_leftshift_bits(chr(ord("a") + j), i + 1)
                )
                player_piece = self.player.piece_at_int(
                    square_to_leftshift_bits(chr(ord("a") + j), i + 1)
                )
                if cpu_piece:
                    board[i][j] = ("b" if self.player_white else "w") + cpu_piece
                elif player_piece:
                    board[i][j] = ("w" if self.player_white else "b") + player_piece

        print("    a   b   c   d   e   f   g   h")
        print("    -----------------------------")
        for i in range(7, -1, -1):
            print(f" {i+1} ", end="")
            for j in range(8):
                piece = board[i][j]
                if piece[0] == "w":
                    print(f" {Player.w_icons[piece[1]]} ", end=" ")  # White piece
                elif piece[0] == "b":
                    print(f" {Player.b_icons[piece[1]]} ", end=" ")  # Black piece
                else:
                    print(" . ", end=" ")  # Empty square
            print(f" {i+1}")
        print("    -----------------------------")
        print("    a   b   c   d   e   f   g   h")
