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
        self.calculate_possible_moves()
        self.selected_pawn = None

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

    def draw_board(self, draw_coordinates=True, ):
        move_str = "Green" if self.move_green else "Blue"
        print(f"Now move: " + move_str, end="\n\n")
        if self.selected_pawn is not None:
            print(f"Pawn selected: " + "TODO")

        for line_num in range(len(self.board)):
            if draw_coordinates:
                print(line_num, end="|")
            for point in self.board[line_num]:
                print(self.shapes[point], end="")
            print("")

        if draw_coordinates:
            print("  ABCDEFGH")

    def calculate_possible_moves(self):
        """
        Returns a list off all possible moves for every pawn in int form
        """
        pass  # TODO

    def calculate_possible_moves_for_pawn(self, pawn):
        """
        Returns all squares that a given pawn can move to
        """
        pass  # TODO

    def is_move_legal(self, move_ints):
        legal_figures = [1, 2, 3, 4] if self.move_green else [5, 6, 7, 8]
        if self.board[move_ints[1], move_ints[0]] not in legal_figures:
            print("Bad figure selected. Select your figure")
            return False
        if self.board[move_ints[3], move_ints[2]] != 0:
            print("Destination field is occupied. Select other field")
            return False

        # TODO(Czy mogę tak się ruszyć? Typy figur; Skoki)

        return True

    def make_move(self, move):
        # i.e. move = "A9B8"
        move = move_str_to_int(move)

        if not self.is_move_legal(move):
            return

        # swap pawn with empty space
        self.board[move[3],
                   move[2]], self.board[move[1],
                                        move[0]] = self.board[move[1],
                                                              move[0]], self.board[move[3],
                                                                                   move[2]]
        self.move_green = not self.move_green
        self.moves_made += 0.5

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
