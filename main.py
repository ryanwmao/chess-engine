import os
import board

def main():
    print("Welcome to Chess!")
    player_color = input("Choose your color (w/b): ").lower()
    while player_color not in ['w', 'b']:
        player_color = input("Please choose 'w' for white or 'b' for black: ").lower()

    difficulty = input("Choose difficulty (e/h): ").lower()
    while difficulty not in ['e', 'h']:
        difficulty = input("Please choose 'e' for easy or 'h' for hard: ").lower()

    depth = int(input("Enter the depth for CPU minimax: "))
    pruning = input("AB Pruning? (y/n): ").lower()
    while pruning not in ['y', 'n']:
        pruning = input("Please choose 'y' or 'n' for pruning: ").lower()

    game = board.Game(player_white=(player_color == 'w'), hard=(difficulty == 'h'))

    white_turn = True

    while True:
        game.pretty_print()
        if game.player_white == white_turn:
            move_input = input("Enter your move (e.g. e2 e4): ")
        else:
            print("CPU is thinking...")
            if pruning:
                move, _ = game.minimax_ab(game.cpu, game.player, depth)
            else:
                move, _ = game.minimax(game.cpu, game.player, depth)
            if move is None:
                print("You won!")
                break
            game.cpu_move(move)
            white_turn = not white_turn
            os.system('clear')
            continue

        try:
            if len(move_input.split()) == 2:
                from_pos, to_pos = move_input.split()
                from_square = board.square_to_leftshift_bits(from_pos[0], int(from_pos[1]))
                to_square = board.square_to_leftshift_bits(to_pos[0], int(to_pos[1]))

                game.player_move((game.player.piece_at_int(from_square), from_square, to_square))
                os.system('clear')
            else:
                from_pos, to_pos, promo_piece = move_input.split()
                from_square = board.square_to_leftshift_bits(from_pos[0], int(from_pos[1]))
                to_square = board.square_to_leftshift_bits(to_pos[0], int(to_pos[1]))

                game.player_move((game.player.piece_at_int(from_square), from_square, to_square, promo_piece))
                os.system('clear')
            white_turn = not white_turn
        except:
            os.system('clear')
            print("Invalid input. Please enter a valid move.")
        

if __name__ == "__main__":
    main()
