import numpy as np

from .utilities import *


class Position:
    def __init__(self):
        self.shapes = {0: u'\u00B7',  # empty place
                       1: '\033[92m' + u'\u25A1' + '\033[0m',  # white/green square
                       2: '\033[92m' + u'\u25B3' + '\033[0m',  # white/green triangle
                       3: '\033[92m' + u'\u25CB' + '\033[0m',  # white/green circle
                       4: '\033[92m' + u'\u25C7' + '\033[0m',  # white/green dimond
                       5: '\033[94m' + u'\u25C6' + '\033[0m',  # black/blue dimond
                       6: '\033[94m' + u'\u25C9' + '\033[0m',  # black/blue circle
                       7: '\033[94m' + u'\u25BC' + '\033[0m',  # black/blue triangle
                       8: '\033[94m' + u'\u25A0' + '\033[0m',  # black/blue square
                       9: u'\u002A',  # Possible move
                       }
        self.board = self.get_starting_board()
        self.moves_made = 0
        self.move_green = True  # Green True, Blue False
        self.is_terminal = False
        self.legal_moves = []
        self.calculate_possible_moves()  # overwrite self.legal_moves
        self.selected_pawn = None
        self.legal_figures = [1, 2, 3, 4]

    @staticmethod
    def get_starting_board():
        empty_middle = np.array([[0, 0, 0, 0, 0, 0, 0, 0] for _ in range(8)])
        starting_white = np.array([[2, 3, 4, 1, 4, 3, 1, 2]])
        starting_black = 9 - starting_white

        starting_board = np.concatenate((starting_black,
                                         empty_middle,
                                         starting_white
                                         ))
        return starting_board

    def draw_board(self, draw_coordinates=True):
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
                print(self.shapes[point], end="")
            print("")

        if draw_coordinates:
            print("  ABCDEFGH")

    def select_pawn(self, position):
        self.board[self.board == 9] = 0

        position_int = position_str_to_int(position)

        if not self.board[position_int[0], position_int[1]] in self.legal_figures:
            raise Exception("wrong figure selected")

        self.selected_pawn = position_int

    def calculate_possible_moves(self):
        """
        Returns a list off all possible moves for every pawn in int form
        """

        pass  # TODO

    def distance_to_closest(self, pawn, direction):
        """
        returns the distance frm pawn to other pawn the closest to the pawn.
        If the pawn is at the edge, returns 0.
        If there is no other pawn up to the border, returns 20
        """

        # TODO
        pass

    def calculate_legal_moves_for_pawn(self, pawn, only_jumps=False):
        """
        Returns all squares that a given pawn can move to
        """
        # i.e. pawn = [9, 4] so it is position_int

        pawn_figure = self.board[pawn[0], pawn[1]]

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

        legal_moves_for_pawn = []

        for direction in directions_to_look_for_moves:
            pawn_distance_to_closest = distance_to_closest(pawn, direction)
            if pawn_distance_to_closest > 1:
                next_place_for_pawn = next_place(pawn, direction)
                if not only_jumps:
                    move = [pawn[0], pawn[1], next_place_for_pawn[0], next_place_for_pawn[1]]

                    legal_moves_for_pawn.append(move)

                if pawn_distance_to_closest > 10:  # there is no other pawn to jump over
                    continue

                closest_other_pawn = next_place(next_place_for_pawn, direction, pawn_distance_to_closest)
                pawn_distance_to_next = distance_to_closest(closest_other_pawn, direction)

                if pawn_distance_to_next > pawn_distance_to_closest:
                    next_place_for_pawn_jump = next_place(pawn, 2 * pawn_distance_to_closest)
                    if next_place_for_pawn_jump is None:  # the board has ended
                        continue

                    # jump is legal
                    move = [pawn[0], pawn[1], next_place_for_pawn_jump[0], next_place_for_pawn_jump[1]]

                    legal_moves_for_pawn.append(move)

        return legal_moves_for_pawn

    def is_move_legal(self, move_ints):
        if self.board[move_ints[0], move_ints[1]] not in self.legal_figures:
            print("Bad figure selected. Select your figure")
            return False
        if not self.board[move_ints[2], move_ints[3]] in [0, 9]:
            print("Destination field is occupied. Select other field")
            return False

        # TODO(Czy mogę tak się ruszyć? Typy figur; Skoki)

        return True

    def make_move(self, move):
        # i.e. move = "A9B8"
        move = move_str_to_int(move)
        # i.e. move = [9, 0, 8, 1]

        if not self.is_move_legal(move):
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

        self.selected_pawn = None
        self.board[self.board == 9] = 0

        self.check_is_terminal()

    def check_is_terminal(self):
        if self.moves_made < 1:
            return False

        if np.all(self.board[0] != 0):
            print("White won!")
            self.is_terminal = True
            return True

        if np.all(self.board[9] != 0):
            print("Black won!")
            self.is_terminal = True
            return True

        return False
