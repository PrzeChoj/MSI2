from random import randint
import math

from Taifho import *

class Node(Position):

    def find_children(self):
        "Zwraca wszystkie dzieci węzła, czyli wszystkie dostępne pozycje z aktualnej planszy"
        if self.is_terminal:
            return []
        return self.legal_moves

    def is_leaf(self):
        "Zwraca True jeśli dziecko jest węzeł nie ma dzieci, czyli gdy nie można wykonać już więcej ruchów z danej pozycji planszy"
        return self.is_terminal

    def result(self):
        "Zwraca wynik gry"
        if not self.is_terminal:
            raise RuntimeError(f"Result called on nonterminal board!")
        if self.winner == self.move_green:
            # It's your turn and you've already won. Should be impossible.
            raise RuntimeError(f"reward called on unreachable board!")
        if self.move_green == (not self.winner):
            return 0  # Your opponent has just won. Bad.
        # The winner is neither True or False
        raise RuntimeError(f"board has unknown winner type {self.winner}")

    def h(self):
        "Wynik gry liczony za pomocą heurystyki h, niekoniecznie na podstawie zakończonej planszy"
        pass

    def h_G(self):
        "Wynik gry liczony za pomocą heurystyki h_G, niekoniecznie na podstawie zakończonej planszy"
        pass

    def find_random_child(self):
        "Znajduje losowe dziecko dla węzła, czyli losową nowa dostępną pozycję z aktualnej planszy"
        if self.is_terminal:
            return None
        return self.legal_moves[randint(0, len(self.legal_moves)-1)]

    def calculate_weight(self):
        "Oblicza wagi M do algorytmu PUCT dla każdego dziecka i zwraca w postaci słownika {dziecko = waga}"
        M = dict()
        K = len(self.legal_moves)
        def distance_from_start():
            "Oblicza różnicę odlegości bierki od początku planszy względem aktualnego stanu a wykonanego ruchu. Zwraca wynik dla wszystkich bierek"
            distance = dict()
            for a in self.legal_moves:
                if self.move_green:
                    distance[a] = - (a[2] - a[0])
                else:
                    distance[a] = a[2] - a[0]
            # ToDo - przeskalować wartości w D do odpowiedniego przedziału
            return distance
        D = distance_from_start()
        for a in self.legal_moves:
            weight = math.exp(1 / K * D[a]) / sum([math.exp(1 / K * D[i]) for i in self.legal_moves])
            M[a] = weight
        return M