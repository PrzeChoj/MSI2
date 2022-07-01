import Taifho
import time
import math
from random import randint
from statistics import mean

from MCTS import MCTS_with_heuristic_h
from Node import Node

# saving to file
with open('results/C_h_experiment.csv', 'a') as f:
    f.write(f'game_id;C_value;G_value;who_won;moves_made;time\n')

C_parameters = [1, math.sqrt(2), 2, 3.5, 5]
max_time = 1.5
steps = 6
G = 2

# Game for UCT + h heuristics

for C in C_parameters:  # for each C parameters
    time_of_games = []
    for j in range(0, 10):  # 10 repeats experiment
        print(f"\nExperiment number {j} for a value of C equal to {C}")
        start_game = time.time()
        position = Node()
        while not position.check_is_terminal(print_who_won=False):
            position.calculate_possible_moves()
            if position.get_actual_player() == 1:  # random starts game
                engine_move_int = randint(0, len(position.get_legal_moves()) - 1)
                position.make_move(Taifho.move_int_to_str(position.get_legal_moves()[engine_move_int]))
            else:  # MCTS turn
                engine_mcts = MCTS_with_heuristic_h(C=C, selection_type="UCT", steps=steps)
                num_of_rollouts = 0
                start_time = time.time()
                while True:
                    engine_mcts.do_rollout(position)
                    num_of_rollouts += 1
                    now_time = time.time()
                    sum_time = now_time - start_time  # this is float number
                    if sum_time + 2 * sum_time / num_of_rollouts > max_time:
                        break
                engine_move_node = engine_mcts.choose_move(position)
                engine_move_move = Taifho.which_move_was_made(position.board, engine_move_node.board)
                engine_move_int = None
                for i in range(len(position.get_legal_moves())):
                    if position.get_legal_moves()[i] == engine_move_move:
                        engine_move_int = i
                position.make_move(Taifho.move_int_to_str(position.get_legal_moves()[engine_move_int]))
        end_game = time.time()
        print(f"{'Random engine' if position.winner else 'MCTS engine'} ({'Green' if position.winner else 'Blue'}) won after {position.moves_made} moves!")
        game_time = end_game - start_game
        time_of_games.append(game_time)
        print(f"Game lasted {game_time} seconds")

        # saving to file
        with open('results/C_h_experiment.csv', 'a') as f:
            f.write(f'{j};{C};NULL;{"Random" if position.winner else "MCTS"};{position.moves_made};{game_time}\n')

        print("\n***********************************************")
    mean_time = mean(time_of_games)
    print(f"\nMean time of game: {mean_time}\n")
