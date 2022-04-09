import numpy as np

class Position:
    def __init__(self):
        self.shapes = {0: u'\u00B7',  # empty place
                       1: u'\u25A1',  # white square
                       2: u'\u25B3',  # white triangle
                       3: u'\u25CB',  # white circle
                       4: u'\u25C7',  # white dimond
                       5: u'\u25C6',  # black dimond
                       6: u'\u25C9',  # black circle
                       7: u'\u25BC',  # black triangle
                       8: u'\u25A0',  # black square
                       }
        self.board = self.get_starting_board()
        self.moves_made = 0
        self.move_white = True  # White True, Black False
        self.is_terminal = False

    def get_starting_board(self):
        empty_middle = np.array([[0, 0, 0, 0, 0, 0, 0, 0] for i in range(8)])
        starting_white = np.array([[2, 3, 4, 1, 4, 3, 1, 2]])
        starting_black = 9 - starting_white

        starting_board = np.concatenate((starting_black,
                                         empty_middle,
                                         starting_white
                                         ))
        return starting_board

    def draw_board(self, draw_coordinates=True):
        move_str = "White" if self.move_white else "Black"
        print(f"Now move: " + move_str, end="\n\n")

        for line_num in range(len(self.board)):
            if draw_coordinates:
                print(line_num, end="|")
            for point in self.board[line_num]:
                print(self.shapes[point], end="")
            print("")

        if draw_coordinates:
            print("  ABCDEFGH")

    def get_possible_moves(self):
        pass

    def is_move_legal(self, move_ints):
        legal_figures = [1, 2, 3, 4] if self.move_white else [5, 6, 7, 8]
        if self.board[move_ints[1], move_ints[0]] not in legal_figures:
            print("Bad figure selected. Select your figure")
            return False
        if self.board[move_ints[3], move_ints[2]] != 0:
            print("Destination field is occupied. Select other field")
            return False

        # TODO(Czy mogę tak się ruszyć? Typy figur; Skoki)

        return True

    def move_str_to_int(self, move_str):
        move_str = move_str.lower()
        move_start_letter = ord(move_str[0]) - 97  # not 96, cos we want a -> 0
        move_start_number = int(move_str[1])
        move_end_letter = ord(move_str[2]) - 97
        move_end_number = int(move_str[3])

        return [move_start_letter, move_start_number, move_end_letter, move_end_number]

    def make_move(self, move):
        # i.e. move = "A9B8"
        move = self.move_str_to_int(move)

        if not self.is_move_legal(move):
            return

        self.board[move[3], move[2]], self.board[move[1], move[0]] = self.board[move[1], move[0]], self.board[
            move[3], move[2]]
        self.move_white = not self.move_white
        self.moves_made += 0.5

        self.check_is_therminal()

    def check_is_therminal(self):
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


