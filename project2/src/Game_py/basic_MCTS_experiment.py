import Taifho
import time

from MCTS import MCTS
from Node import Node



position = Node()

position.draw_board()

engine_mcts = MCTS(selection_type="UCT")

max_time = 1
num_of_rollouts = 0
start_time = time.time()
while True:
    engine_mcts.do_rollout(position)  # engine is unable to do even one rollout...
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
