import numpy as np
from copy import deepcopy
from itertools import chain

from .utilities import *


class Position:
    def __init__(self):
        self.shapes = {0: u'\u00B7',  # empty place
                       1: '\033[92m' + u'\u25A1' + '\033[0m',  # Player1/green square
                       2: '\033[92m' + u'\u25B3' + '\033[0m',  # Player1/green triangle
                       3: '\033[92m' + u'\u25CB' + '\033[0m',  # Player1/green circle
                       4: '\033[92m' + u'\u25C7' + '\033[0m',  # Player1/green dimond
                       5: '\033[94m' + u'\u25C6' + '\033[0m',  # Player2/blue dimond
                       6: '\033[94m' + u'\u25C9' + '\033[0m',  # Player2/blue circle
                       7: '\033[94m' + u'\u25BC' + '\033[0m',  # Player2/blue triangle
                       8: '\033[94m' + u'\u25A0' + '\033[0m',  # Player2/blue square
                       9: u'\u002A',  # Possible move
                       }
        self.board = self.get_starting_board()
        self.moves_made = 0
        self.move_green = True  # Green True, Blue False
        self.is_terminal = False
        self.legal_figures = [1, 2, 3, 4]
        self.legal_moves = []
        self.winner = None
        self.calculate_possible_moves()  # overwrite self.legal_moves
        self.selected_pawn = None

    def get_actual_player(self):
        """
        Zwraca 1 gdy aktualny ruch należy do zielonego gracza, 2 gdy do niebieskiego
        """
        if self.move_green:
            return 1
        else:
            return 2

    @staticmethod
    def get_starting_board():
        """
        Zwraca startową pozycję. Funkcja statyczna
        """
        empty_middle = np.array([[0, 0, 0, 0, 0, 0, 0, 0] for _ in range(8)])
        starting_white = np.array([[2, 3, 4, 1, 4, 3, 1, 2]])
        starting_black = 9 - starting_white

        starting_board = np.concatenate((starting_black,
                                         empty_middle,
                                         starting_white
                                         ))
        return starting_board

    def draw_board(self, draw_coordinates=True):
        """
        Ryzuje pozycje. Bierze pod uwagę czyj jest ruch. Jeśli jakaś bierka jest
            wybrana (czyli self.selected_pawn is not None), to rysuje * w miejscu gdzie ona może się ruszyć
        """
        if self.winner is None:
            move_str = "Green" if self.move_green else "Blue"
            print(f"Now move: " + move_str, end="\n\n")

        self.board[self.board == 9] = 0

        board = self.board

        if self.selected_pawn is not None:
            print(f"Pawn selected: " + position_int_to_str(self.selected_pawn))

            legal_moves_for_pawn = self.calculate_legal_moves_for_pawn(self.selected_pawn)
            for pos in legal_moves_for_pawn:
                board[pos[2], pos[3]] = 9  # change 0 into 9

        for line_num in range(len(board)):
            if draw_coordinates:
                print(line_num, end="|")
            for point in board[line_num]:
                print(self.shapes[point], end="  ")
            print("")

        if draw_coordinates:
            print("  A  B  C  D  E  F  G  H")

    def select_pawn(self, position_str=None):
        """
        zaznacza wybraną bierkę. Przy rysowaniu self.draw_board będą oznaczone jej możliwe ruchy
        """
        self.board[self.board == 9] = 0

        if position_str is None:
            self.selected_pawn = None
            return None  # TODO(Make sure I can return None here)

        position_int = position_str_to_int(position_str)

        if not self.board[position_int[0], position_int[1]] in self.legal_figures:
            print("Wrong figure selected. Select your figure")
            return False

        self.selected_pawn = position_int
        return True

    def calculate_possible_moves(self):
        """
        Tworzy listę wszystkich możliwych ruchów dla wszystkich bierek w stylu int. Nie zwraca nic.
        """
        if self.winner is not None:
            self.legal_moves = []
        legal_moves = []
        for fig in self.legal_figures:
            pawns = np.where(self.board == fig)
            for i in range(0, len(pawns)):
                legal_moves_for_pawn = self.calculate_legal_moves_for_pawn([pawns[0][i], pawns[1][i]])
                legal_moves.append(legal_moves_for_pawn)
        self.legal_moves = list(chain.from_iterable(legal_moves))

    def distance_to_closest(self, pawn, direction, board=None):
        """
        Zwraca jak duża jest odległość z bierki do innej w wybranym kierunku.
        Jeśli następne pole w danym kierunku jest poza planszą, zwróć 0.
        Jeśli w danym kierunku niema innych bierek aż do ściany, ale następne pole w danym kierunku
            NIE jest poza planszą, zwróć 20.

        możliwe kierunki direction: 0 - up, 1 - up-right 2 - right, ..., 7 - up-left
        """
        if board is None:
            board = copy(self.board)
        if next_place(pawn, direction) is None:
            return 0
        for step in range(1, 11):
            new_place = next_place(pawn, direction, step)
            if new_place is None:
                return 20
            if board[new_place[0], new_place[1]] not in [0, 9]:
                return step

    def calculate_legal_moves_for_pawn(self, pawn, only_jumps=False, board=None, already_found_moves=[], org_pawn=None):
        """
        Zwraca listę możliwych ruchów dla danej bierki w stylu int. Lista ta zawiera cały ruch, czyli
            współrzędne punktu jak i pola docelowe.

        Wywułuje się rekurencyjnie dla obliczania skoków. Wtedy only_jumps == True, board jest zmodyfikowana,
            already_found_moves jest listą róchów już znalezionych. Jeśli trafił w pole w którym już był,
            to przerywa rekurencję. org_pawn to współrzędne z którego pionek zaczął skoki i
            jest inny niż None w rekurencji. Użyty jest do poprawnego zapisu.

        Jest nieotestowana, bo trzeba najpierw zaimplementować funkcję self.distance_to_closest() oraz
            funkcję next_place(), co jeszcze się nie stało.
        """
        # i.e. pawn = [9, 4] so it is position_int

        if board is None:
            board = copy(self.board)
        if org_pawn is None:
            org_pawn = pawn

        pawn_figure = board[pawn[0], pawn[1]]

        if pawn_figure not in self.legal_figures:
            raise Exception("wrong figure selected")

        directions_to_look_for_moves = []  # 0 - up, 1 - up-right 2 - right, ..., 7 - up-left
        if pawn_figure in [1, 8]:
            directions_to_look_for_moves = [0, 2, 4, 6]
        elif pawn_figure in [3, 6]:
            directions_to_look_for_moves = [0, 1, 2, 3, 4, 5, 6, 7]
        elif pawn_figure in [4, 5]:
            directions_to_look_for_moves = [1, 3, 5, 7]
        elif pawn_figure in [2]:
            directions_to_look_for_moves = [1, 4, 7]
        elif pawn_figure in [7]:
            directions_to_look_for_moves = [0, 3, 5]

        legal_moves_for_pawn = copy(already_found_moves)  # for the recursive, there are

        for direction in directions_to_look_for_moves:
            pawn_distance_to_closest = self.distance_to_closest(pawn, direction, board)
            if pawn_distance_to_closest > 1 and not only_jumps:  # a pawn can move to this direction, there is no wall nor other pawn on the very next tile
                next_place_for_pawn = next_place(pawn, direction)
                move = [pawn[0], pawn[1], next_place_for_pawn[0], next_place_for_pawn[1]]

                legal_moves_for_pawn.append(move)

            if pawn_distance_to_closest >= 1:  # there may be possible jump in this direction
                if pawn_distance_to_closest == 20:  # there is no other pawn to jump over
                    continue
                closest_other_pawn = next_place(pawn, direction, pawn_distance_to_closest)
                if closest_other_pawn is None:  # the board has ended, cannot jump out of bounds
                    continue
                pawn_distance_to_next = self.distance_to_closest(closest_other_pawn, direction, board)  # is there another pawn that will prevent us from jumping?

                if pawn_distance_to_next > pawn_distance_to_closest:
                    next_place_for_pawn_jump = next_place(pawn, direction, 2 * pawn_distance_to_closest)
                    if next_place_for_pawn_jump is None:  # the board has ended, cannot jump out of bounds; this happens when pawn_distance_to_next == 20
                        continue

                    # jump is legal
                    move = [org_pawn[0], org_pawn[1], next_place_for_pawn_jump[0], next_place_for_pawn_jump[1]]

                    if org_pawn[0] == move[2] and org_pawn[1] == move[3]:
                        continue  # go to the next direction, this place was the starting position
                    to_add = True
                    for move_i in already_found_moves:
                        if move_i[2] == move[2] and move_i[3] == move[3]:
                            to_add = False
                            break  # go to the next direction, this place was already found

                    if not to_add:  # go to the next direction, this place was already found
                        continue

                    legal_moves_for_pawn.append(move)

                    # next jumps recursive:
                    board_after_move = deepcopy(board)
                    board_after_move[pawn[0], pawn[1]], board_after_move[next_place_for_pawn_jump[0], next_place_for_pawn_jump[1]] = board_after_move[next_place_for_pawn_jump[0], next_place_for_pawn_jump[1]], board_after_move[pawn[0], pawn[1]]

                    next_jumps_for_pawn = self.calculate_legal_moves_for_pawn(next_place_for_pawn_jump, only_jumps=True,
                                                                              board=board_after_move,
                                                                              already_found_moves=legal_moves_for_pawn,
                                                                              org_pawn=org_pawn)

                    for i in range(len(legal_moves_for_pawn), len(next_jumps_for_pawn)):  # the next_jumps_for_pawn starts with legal_moves_for_pawn and then there are new moves
                        legal_moves_for_pawn.append(next_jumps_for_pawn[i])

        return legal_moves_for_pawn

    def is_move_legal(self, move_ints):
        """
        Funkcja wywoływana gdy user będzie próbował się ruszyć. Sprowadza się do sprawdzenia, czy move_ints jest w self.legal_moves
        """
        if self.board[move_ints[0], move_ints[1]] not in self.legal_figures:
            print("Wrong figure selected. Select your figure")
            return False
        if self.board[move_ints[2], move_ints[3]] not in [0, 9]:
            print("Destination field is occupied. Select other field")
            return False
        return move_ints in self.legal_moves

    def make_move(self, move_str):
        """
        Zamienia bierkię i puste pole na planszy
        """
        move = move_str_to_int(move_str)

        if not self.is_move_legal(move):
            print("The selected move is not allowed. Select other move")
            return

        # swap pawn with empty space
        self.board[move[2],
                   move[3]], self.board[move[0],
                                        move[1]] = self.board[move[0],
                                                              move[1]], self.board[move[2],
                                                                                   move[3]]
        self.move_green = not self.move_green
        self.legal_figures = [1, 2, 3, 4] if self.move_green else [5, 6, 7, 8]
        self.moves_made += 0.5
        self.calculate_possible_moves()

        self.selected_pawn = None
        self.board[self.board == 9] = 0  # usuwa stare zapisy możliwych ruchów

        self.check_is_terminal(print_who_won=False)

    def check_is_terminal(self, print_who_won=True):
        """
        Sprawdza, czy któremuś graczowi udało się już wygrać
        """
        if self.moves_made < 1:
            return False
        if np.all(np.logical_and(self.board[0] != 0, self.board[0] != 9)):
            if print_who_won:
                print("\nGreen won!")
            self.is_terminal = True
            self.winner = True
            return True
        if np.all(np.logical_and(self.board[9] != 0, self.board[9] != 9)):
            if print_who_won:
                print("\nBlue won!")
            self.is_terminal = True
            self.winner = False
            return True
        return False
