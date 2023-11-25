import os
import board

def main():
    print("Welcome to Chess!")
    player_color = input("Choose your color (w/b): ").lower()
    while player_color not in ['w', 'b']:
        player_color = input("Please choose 'w' for white or 'b' for black: ").lower()

    depth = int(input("Enter the depth for CPU minimax: "))

    game = board.Game(player_white=(player_color == 'w'))

    white_turn = True

    while True:
        os.system('clear')
        game.pretty_print()
        if game.player_white == white_turn:
            move_input = input("Enter your move (from_col from_row to_col to_row): ")
        else:
            print("CPU is thinking...")
            move, _ = game.minimax(game.cpu, game.player, depth)
            game.cpu_move(move)
            white_turn = not white_turn
            continue

        try:
            if len(move_input.split()) == 4:
                from_col, from_row, to_col, to_row = move_input.split()
                from_square = board.square_to_leftshift_bits(from_col, int(from_row))
                to_square = board.square_to_leftshift_bits(to_col, int(to_row))

                game.player_move((game.player.piece_at_int(from_square), from_square, to_square))
            else:
                from_col, from_row, to_col, to_row, promo_piece = move_input.split()
                from_square = board.square_to_leftshift_bits(from_col, int(from_row))
                to_square = board.square_to_leftshift_bits(to_col, int(to_row))

                game.player_move((game.player.piece_at_int(from_square), from_square, to_square, promo_piece))
            white_turn = not white_turn
        except ValueError:
            print("Invalid input. Please enter a valid move.")
        

if __name__ == "__main__":
    main()
