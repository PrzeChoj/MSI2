from random import randint
import time

from Taifho import *

# ToDo - zrobić sprawdzanie czy podane argumenty figury i ruchu są poprawne i można zrobić zrzutowanie!!! Tutaj lub w pakiecie!
# ToDo -popakować to w funkcje, aby było czytelniej

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
        print("\n\t\t\t\tStarting the game...\n")
        print("[1] Green (starting player)")
        print("[2] Blue")
        while color == "":
            try:
                color = int(input("\nSelect your pawns' color: "))
            except:
                print("\nWrong value.")
                color = ""
                continue
            if color not in [1, 2]:
                print("\nWrong numer.")
                color = ""
        position = Position()
        end_game = position.check_is_terminal()
        while not end_game:
            print("\n")
            position.draw_board()
            if position.get_actual_player() == color:
                while selected_pawn == "":
                    selected_pawn = input("\nSelect pawn to make a move by entering letter and number denoting the position (np. A1): ")
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
                            print("\nWrong value.")
                            continue
                        if move_choice not in [1, 2]:
                            print("\nWrong numer.")
                            move_choice = ""
                            continue
                        if move_choice == 1:
                            selected_pawn = ""
                            break
                        else:
                            while position_to_move == "":
                                position_to_move = input("Select new position for pawn: ")
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
                engine_move = randint(0, len(position.legal_moves)-1)
                position.make_move(move_int_to_str(position.legal_moves[engine_move]))
            end_game = position.check_is_terminal()
            selected_pawn = ""
            position_to_move = ""
            move_choice = ""
            move = []

    elif choice == "q":

        print("\t*************************************************")
        print("\n\t\t\t\t\t\tGoodbye!\n")
        print("\t*************************************************")

    else:

        print("Your choice is wrong.\n")
