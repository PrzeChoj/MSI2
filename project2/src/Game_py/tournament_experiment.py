import Taifho
import time
from random import randint, seed
from statistics import mean

import numpy as np
import math

from MCTS import MCTS_with_heuristic_h, MCTS_with_heuristic_h_G
from Node import Node

# saving to file
with open('results/1_tournament_UCT_with_both_heuristics_experiment.csv', 'a') as f:
    f.write(f'game_id;seed;who_Green;who_Blue;who_won;moves_made;time\n')

seed_values = [123, 245, 456, 786, 999, 567, 582, 11, 765, 66]
time_of_games = []
for j in range(0, 10):  # 10 repeats experiment
    print(f"\nExperiment number {j}")
    start_game = time.time()
    position = Node()
    position.draw_board()
    seed(seed_values[j])  # set seed
    first_player = randint(1, 2)  # choose who starts game
    print(f"Game starts {'UCT + h C=sqrt(2)' if first_player==1 else 'UCT + h_G 20'}")
    while not position.check_is_terminal(print_who_won=False):
        position.calculate_possible_moves()
        if position.get_actual_player() == first_player:
            engine_mcts = MCTS_with_heuristic_h(C=math.sqrt(2), selection_type="UCT", steps=6)
        else:
            engine_mcts = MCTS_with_heuristic_h_G(C=3.5, selection_type="UCT", steps=6, G=20)
        num_of_rollouts = 0
        start_time = time.time()
        while True:
            engine_mcts.do_rollout(position)
            num_of_rollouts += 1
            now_time = time.time()
            sum_time = now_time - start_time
            if sum_time + 2 * sum_time / num_of_rollouts > 2:
                break
        engine_move_node = engine_mcts.choose_move(position)
        engine_move_move = Taifho.which_move_was_made(position.board, engine_move_node.board)
        engine_move_int = None
        for i in range(len(position.get_legal_moves())):
            if position.get_legal_moves()[i] == engine_move_move:
                engine_move_int = i
        position.make_move(Taifho.move_int_to_str(position.get_legal_moves()[engine_move_int]))
        position.draw_board()
    end_game = time.time()
    print(f"{'MCTS + UCT + h heuristics C=sqrt(2)' if position.winner==(first_player==1) else 'MCTS + UCT + h_G heuristics 20'} ({'Green' if position.winner==(first_player==1) else 'Blue'}) won after {position.moves_made} moves!")
    game_time = end_game - start_game
    time_of_games.append(game_time)
    print(f"Game lasted {game_time} seconds")

    # saving to file
    with open('results/1_tournament_UCT_with_both_heuristics_experiment.csv', 'a') as f:
        f.write(f'{j};{seed_values[j]};{"UCT+h" if first_player==1 else "UCT+h_G"};{"UCT+h_G" if first_player==1 else "UCT+h"};{"UCT+h" if position.winner==(first_player==1) else "UCT+h_G"};{position.moves_made};{game_time}\n')

    print("\n***********************************************")
mean_time = mean(time_of_games)
print(f"\nMean time of game: {mean_time}\n")
