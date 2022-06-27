from random import randint
import time
import re

from Taifho import *

print("\t*************************************************")
print("\n\t\t\t\tWelcome to Taifho Game!\n")
print("\t*************************************************")

choice = " "

while choice != "q":

    print("[1] Start Game!")
    print("[q] Quit.")
    choice = input("\nSelect what do you want to do: ")

    if choice == "1":

        color = ""
        selected_pawn = ""
        position_to_move = ""
        move_choice = ""
        move = []
        engine = ""
        print("\n\t\t\t\tStarting the game...\n")
        print("[1] Random engine")
        print("[2] MCTS + UCT engine")  # TODO(zmienić nazwy zgodnie z planowanymi silnikami)
        print("[3] MCTS + PUCT engine")
        while engine == "":
            try:
                engine = int(input("\nSelect engine type of your enemy in Taifho: "))
            except:
                print("\nWrong value selected.")
                engine = ""
                continue
            if engine not in [1, 2, 3]:  # TODO(ustawić tyle liczb ile trzeba)
                print("\nWrong number selected.")
                engine = ""
                continue
        print("[1] Green (starting player)")
        print("[2] Blue")
        while color == "":
            try:
                color = int(input("\nSelect your pawns' color: "))
            except:
                print("\nWrong value selected.")
                color = ""
                continue
            if color not in [1, 2]:
                print("\nWrong numer selected.")
                color = ""
        position = Position()
        end_game = position.check_is_terminal()
        while not end_game:
            print("\n")
            position.draw_board()
            if position.get_actual_player() == color:
                while selected_pawn == "":
                    selected_pawn = input("\nSelect pawn to make a move by entering letter and number denoting the position (eg. A1): ")
                    if not re.match(re.compile("^([A-H][0-9])"), selected_pawn):
                        print("The wrong value has been entered. Select your figure")
                        selected_pawn = ""
                        continue
                    if not position.select_pawn(selected_pawn):
                        selected_pawn = ""
                        continue
                    print("\n")
                    position.draw_board()
                    print("\n")
                    print("[1] Change selected pawn")
                    print("[2] Make a move")
                    while True:
                        try:
                            move_choice = int(input("Select what do you want to do now: "))
                        except:
                            print("\nWrong value selected.")
                            continue
                        if move_choice not in [1, 2]:
                            print("\nWrong number selected.")
                            move_choice = ""
                            continue
                        if move_choice == 1:
                            position.select_pawn()  # unselect pawn
                            position.draw_board()
                            selected_pawn = ""
                            break
                        else:
                            while position_to_move == "":
                                position_to_move = input("Select new position for pawn: ")
                                if not re.match(re.compile("^([A-H][0-9])"), position_to_move):
                                    print("The wrong value has been entered. Enter proper position")
                                    position_to_move = ""
                                    continue
                                move = [*position_str_to_int(selected_pawn), *position_str_to_int(position_to_move)]
                                if not position.is_move_legal(move):
                                    print("Wrong position. Try again.")
                                    position_to_move = ""
                                    move = []
                            position.make_move(selected_pawn + position_to_move)
                        break
            else:
                # na ten moment napisane dla losowego silnika
                time.sleep(1)
                position.calculate_possible_moves()
                if engine == 1:
                    engine_move = randint(0, len(position.legal_moves)-1)
                    print("\n\nEngine moved " + move_int_to_str(position.legal_moves[engine_move])[0] +
                          move_int_to_str(position.legal_moves[engine_move])[1] + " goes to " +
                          move_int_to_str(position.legal_moves[engine_move])[2] +
                          move_int_to_str(position.legal_moves[engine_move])[3], end="")
                    position.make_move(move_int_to_str(position.legal_moves[engine_move]))
                elif engine == 2:
                    pass  # TODO()
                elif engine == 3:
                    pass  # TODO()
            end_game = position.check_is_terminal()
            selected_pawn = ""
            position_to_move = ""
            move_choice = ""
            move = []

        position.draw_board()  # Draw board after the winning

    elif choice == "q":

        print("\t*************************************************")
        print("\n\t\t\t\t\t\tGoodbye!\n")
        print("\t*************************************************")

    else:

        print("Your choice is wrong.\n")
