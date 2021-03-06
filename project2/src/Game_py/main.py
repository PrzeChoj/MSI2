from random import randint
import time
import re
import math

import Taifho

from MCTS import MCTS, MCTS_with_heuristic_h, MCTS_with_heuristic_h_G
from Node import make_Node_from_Position

print("\t*************************************************")
print("\n\t\t\t\tWelcome to Taifho Game!\n")
print("\t*************************************************")

choice = " "

while choice != "q":

    print("[1] Start Game!")
    print("[2] Debug")
    print("[q] Quit.")
    choice = input("\nSelect what do you want to do: ")

    if choice == "1" or choice == "2":

        color = ""
        selected_pawn = ""
        position_to_move = ""
        move_choice = ""
        move = []
        engine = ""
        C = ""
        max_time = ""
        G = ""
        steps = ""
        print("\n\t\t\t\tStarting the game...\n")
        print("[1] Random engine")
        print("[2] MCTS + UCT engine (as expected: it does not work)")
        print("[3] MCTS + PUCT engine (as expected: it does not work)")
        print("[4] MCTS + UCT engine + h heuristic")
        print("[5] MCTS + UCT engine + h_G heuristic")
        print("[6] MCTS + PUCT engine + h heuristic")
        print("[7] MCTS + PUCT engine + h_G heuristic")
        while engine == "":
            try:
                engine = int(input("\nSelect engine type of your enemy in Taifho: "))
            except:
                print("\nWrong value selected.")
                engine = ""
                continue
            if engine not in [1, 2, 3, 4, 5, 6, 7]:
                print("\nWrong number selected.")
                engine = ""
                continue
        if engine != 1:
            while C == "":
                try:
                    C_str = input("\nSelect C value for UCT (press enter for default, sqrt(2)): ")
                    C = math.sqrt(2) if C_str == "" else float(C_str)
                    if C <= 0:
                        raise Exception("C has to be strictly bigger than 0")
                except:
                    print("\nWrong value selected. C has to be positive real number.")
                    C = ""
                    continue
            while max_time == "":
                try:
                    max_time_str = input("\nSelect maximum time for each move for engine (in seconds) (press enter for default, 10): ")
                    max_time = 10 if max_time_str == "" else float(max_time_str)
                    if max_time <= 0:
                        raise Exception("max_time has to be strictly bigger than 0")
                except:
                    print("\nWrong value selected. maximum time has to be positive real number.")
                    max_time = ""
                    continue
        if engine in [5, 7]:
            while G == "":
                try:
                    G_str = input("\nSelect G value for UCT (press enter for default, 20): ")
                    G = 20 if G_str == "" else float(G_str)
                    if G <= 1:
                        raise Exception("G has to be strictly bigger than 1")
                except:
                    print("\nWrong value selected. G has to be real number strictly bigger tan 1.")
                    G = ""
                    continue
        if engine > 3:
            while steps == "":
                try:
                    steps_str = input("\nSelect maximum number of steps of simulation for engine (press enter for default, 6): ")
                    steps = 6 if steps_str == "" else float(steps_str)
                    if steps <= 0:
                        raise Exception("steps has to be strictly bigger than 0")
                except:
                    print("\nWrong value selected. steps has to be positive integer.")
                    steps = ""
                    continue
        print("\n[1 or enter] Green (starting player)")
        print("[2] Blue")
        while color == "":
            try:
                color_str = input("\nSelect your pawns' color: ")
                color = 1 if color_str == "" else int(color_str)
            except:
                print("\nWrong value selected.")
                color = ""
                continue
            if color not in [1, 2]:
                print("\nWrong numer selected.")
                color = ""
        position = Taifho.Position()
        print("\n\n\t\t\t\t\tStart Game!")
        while not position.check_is_terminal(print_who_won=False):
            print("\n")
            position.draw_board()
            if position.get_actual_player() == color:
                while selected_pawn == "":
                    selected_pawn = input(f"\nSelect pawn to make a move by entering letter and number denoting the position (eg. {Taifho.position_int_to_str([position.get_legal_moves()[0][0], position.get_legal_moves()[0][1]])}): ")
                    if not re.match(re.compile("^([a-hA-H][0-9])"), selected_pawn):
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
                    print("[2 or enter] Make a move")
                    while True:
                        try:
                            move_choice_org = input("Select what do you want to do now: ")
                            move_choice = 2 if move_choice_org == "" else int(move_choice_org)
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
                                if not re.match(re.compile("^([a-hA-H][0-9])"), position_to_move):
                                    print("The wrong value has been entered. Enter proper position")
                                    position_to_move = ""
                                    continue
                                move = [*Taifho.position_str_to_int(selected_pawn), *Taifho.position_str_to_int(position_to_move)]
                                if not position.is_move_legal(move):
                                    print("Wrong position. Try again.")
                                    position_to_move = ""
                                    move = []
                            position.make_move(selected_pawn + position_to_move)
                        break
            else:
                position.calculate_possible_moves()
                print("\nEngine thinks: ...")
                if engine == 1:
                    time.sleep(1)
                    engine_move_int = randint(0, len(position.get_legal_moves())-1)
                else:
                    node = make_Node_from_Position(position)
                    engine_mcts = None
                    if engine == 2:
                        engine_mcts = MCTS(C=C, selection_type="UCT")
                    elif engine == 3:
                        engine_mcts = MCTS(C=C, selection_type="PUCT")
                    elif engine == 4:
                        engine_mcts = MCTS_with_heuristic_h(C=C, selection_type="UCT", steps=steps)
                    elif engine == 5:
                        engine_mcts = MCTS_with_heuristic_h_G(C=C, selection_type="UCT", steps=steps, G=G)
                    elif engine == 6:
                        engine_mcts = MCTS_with_heuristic_h(C=C, selection_type="PUCT", steps=steps)
                    elif engine == 7:
                        engine_mcts = MCTS_with_heuristic_h_G(C=C, selection_type="PUCT", steps=steps, G=G)
                    num_of_rollouts = 0
                    start_time = time.time()
                    while True:
                        engine_mcts.do_rollout(node)
                        num_of_rollouts += 1
                        now_time = time.time()
                        sum_time = now_time - start_time  # this is float number
                        if sum_time + 2 * sum_time/num_of_rollouts > max_time:
                            break
                    if choice == "2":
                        print(f"branching factor = {len(engine_mcts.children[node])}")
                        print(f"num_of_rollouts = {num_of_rollouts}")
                    engine_move_node = engine_mcts.choose_move(node)
                    engine_move_move = Taifho.which_move_was_made(node.board, engine_move_node.board)
                    engine_move_int = None
                    for i in range(len(position.get_legal_moves())):
                        if position.get_legal_moves()[i] == engine_move_move:
                            engine_move_int = i

                print("\nEngine moved " + Taifho.move_int_to_str(position.get_legal_moves()[engine_move_int])[0] +
                      Taifho.move_int_to_str(position.get_legal_moves()[engine_move_int])[1] + " goes to " +
                      Taifho.move_int_to_str(position.get_legal_moves()[engine_move_int])[2] +
                      Taifho.move_int_to_str(position.get_legal_moves()[engine_move_int])[3], end="")
                position.make_move(Taifho.move_int_to_str(position.get_legal_moves()[engine_move_int]))
            selected_pawn = ""
            position_to_move = ""
            move_choice = ""
            move = []

        print(f"\n{'You' if position.winner == (color == 1) else 'MCTS'} ({'Green' if position.winner else 'Blue'}) won after {position.moves_made} moves!\n")
        position.draw_board()  # Draw board after the winning

        print("\n\t*************************************************")
        print("\n\t\t\t\t\t\tEnd Game!\n")
        print("\t*************************************************\n")

    elif choice == "q":

        print("\t*************************************************")
        print("\n\t\t\t\t\t\tGoodbye!\n")
        print("\t*************************************************")

    else:

        print("Your choice is wrong.\n")
