import Taifho
import copy
import numpy as np
import time

from MCTS import MCTS, MCTS_with_heuristic_h, MCTS_with_heuristic_h_G
from Node import Node, make_Node_from_Position

position = Node()

position.board = np.array([[0,0,0,0,0,0,0,0],
                           [0,0,0,0,0,0,0,0],
                           [0,0,0,0,0,0,0,0],
                           [0,0,0,0,0,0,0,0],
                           [0,0,0,0,0,0,0,0],
                           [0,0,0,0,0,0,0,0],
                           [7,6,5,0,0,0,0,0],
                           [3,1,2,0,0,0,0,0],
                           [2,4,4,0,0,0,0,0],
                           [0,3,1,8,5,6,8,7]])
position.calculate_possible_moves()
position.moves_made = 20

position.draw_board()

engine_mcts = MCTS_with_heuristic_h(selection_type="UCT", steps=2)

max_time = 60  # TODO(Edit here)
num_of_rollouts = 0
start_time = time.time()
while True:
    engine_mcts.do_rollout(position)
    num_of_rollouts += 1
    now_time = time.time()
    sum_time = now_time - start_time  # this is float number
    if sum_time + 2 * sum_time / num_of_rollouts > max_time:
        break
print(f"branching factor = {len(engine_mcts.children[position])}")
print(f"num_of_rollouts = {num_of_rollouts}")
engine_move_node = engine_mcts.choose_move(position)
engine_move_move = Taifho.which_move_was_made(position.board, engine_move_node.board)
engine_move_int = None
for i in range(len(position.legal_moves)):
    if position.legal_moves[i] == engine_move_move:
        engine_move_int = i

print("\nEngine moved " + Taifho.move_int_to_str(position.legal_moves[engine_move_int])[0] +
      Taifho.move_int_to_str(position.legal_moves[engine_move_int])[1] + " goes to " +
      Taifho.move_int_to_str(position.legal_moves[engine_move_int])[2] +
      Taifho.move_int_to_str(position.legal_moves[engine_move_int])[3])
position.make_move(Taifho.move_int_to_str(position.legal_moves[engine_move_int]))

position.draw_board()
