from random import randint
import math
import numpy as np

from Taifho import *


class Node(Position):
    def find_children(self):
        """Zwraca wszystkie dzieci węzła, czyli wszystkie dostępne pozycje z aktualnej planszy"""
        if self.is_terminal:
            return []
        return self.legal_moves

    def is_leaf(self):
        """Zwraca True jeśli dziecko jest węzeł nie ma dzieci, czyli gdy nie można wykonać już więcej ruchów z danej pozycji planszy"""
        return self.is_terminal

    def result(self):
        """Zwraca wynik gry. 0-przegrana, 1-wygrana"""
        if not self.is_terminal:
            raise RuntimeError(f"Result called on nonterminal board!")
        if self.move_green == (not self.winner):
            return 0  # Your opponent has just won. Bad.
        if self.winner == self.move_green:
            # It's your turn and you've won. Means the opponent moved for your win.
            return 1
        # The winner is neither True nor False
        raise RuntimeError(f"board has unknown winner type {self.winner}")

    def h(self):
        """Wynik gry liczony za pomocą heurystyki h, niekoniecznie na podstawie zakończonej planszy"""
        sum_player = 0
        sum_enemy = 0
        if self.legal_figures == [1, 2, 3, 4]:
            for fig in self.legal_figures:
                pawns = np.where(self.board == fig)
                for i in range(0, len(pawns)):
                    sum_player += 9 - pawns[0][i]  # It is right order of 0 and i. The numpy works this way.
        else:
            for fig in [5, 6, 7, 8]:
                pawns = np.where(self.board == fig)
                for i in range(0, len(pawns)):
                    sum_enemy += pawns[0][i]
        return sum_player - sum_enemy

    def h_G(self, G=2):
        """Wynik gry liczony za pomocą heurystyki h_G, niekoniecznie na podstawie zakończonej planszy"""
        if G <= 1:
            raise Exception("Wrong G parameter is selected. Should be G > 1")
        sum_player = 0
        sum_enemy = 0
        if self.legal_figures == [1, 2, 3, 4]:
            for fig in self.legal_figures:
                pawns = np.where(self.board == fig)
                for i in range(0, len(pawns)):
                    sum_player += (9 - pawns[0][i]) * math.log(9 + G) / math.log((9 - pawns[0][i]) + G)
        else:
            for fig in [5, 6, 7, 8]:
                pawns = np.where(self.board == fig)
                for i in range(0, len(pawns)):
                    sum_enemy += pawns[0][i] * math.log(9 + G) / math.log(pawns[0][i] + G)
        return sum_player - sum_enemy

    def find_random_child(self):
        """Znajduje losowe dziecko dla węzła, czyli losową nowa dostępną pozycję z aktualnej planszy"""
        if self.is_terminal:
            return None
        return self.legal_moves[randint(0, len(self.legal_moves)-1)]

    def calculate_weight(self):
        """Oblicza wagi M do algorytmu PUCT dla każdego dziecka i zwraca w postaci słownika {dziecko = waga}"""
        M = dict()
        K = len(self.legal_moves)

        def distance_from_start():
            """Oblicza różnicę odlegości bierki od początku planszy względem aktualnego stanu a wykonanego ruchu. Zwraca wynik dla wszystkich bierek"""
            distance = dict()
            for a in self.legal_moves:
                if self.move_green:
                    distance[a] = - (a[2] - a[0])
                else:
                    distance[a] = a[2] - a[0]
            max_d = max(distance.values())
            min_d = min(distance.values())
            for a in self.legal_moves:
                distance[a] = (distance[a] - min_d) / (max_d - min_d) * (1-1/K - 1/K**3) + 1/K**3  # Scale distance into [1/K**3, 1-1/K]
            return distance
        D = distance_from_start()
        for a in self.legal_moves:
            weight = math.exp(1 / K * D[a]) / sum([math.exp(1 / K * D[i]) for i in self.legal_moves])
            M[a] = weight
        return M


def make_Node_from_Position(position):
    node = Node()

    node.board = position.board
    node.moves_made = position.moves_made
    node.move_green = position.move_green
    node.is_terminal = position.is_terminal
    node.legal_figures = position.legal_figures
    node.legal_moves = position.legal_moves
    node.winner = position.winner
    node.selected_pawn = position.selected_pawn

    return node
