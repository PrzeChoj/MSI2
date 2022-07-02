import math
from collections import defaultdict

from Taifho import move_int_to_str, which_move_was_made
from Node import Node, make_Node_from_Position


class MCTS:
    def __init__(self, C=math.sqrt(2), selection_type="UCT"):
        self.C = C
        self.Q = defaultdict(int)  # total reward of each node
        self.N = defaultdict(int)  # number of visits for each node
        self.children = dict()  # children for each node
        self.selection_type = selection_type

    def choose_move(self, node) -> Node:
        """
        Wybiera najlepszego następcę węzła -> wybiera następny ruch w grze
        """
        if node.is_leaf():
            raise RuntimeError(f"choose called on leaf node. No more moves available!")

        if node not in self.children:
            return node.find_random_child()

        def score(a):
            if self.N[a] == 0:
                return float("-inf")  # avoid unseen moves
            return self.Q[a] / self.N[a]  # average reward

        return max(self.children[node], key=score)

    def do_rollout(self, node) -> None:
        """
        Dodanie do drzewa jednej warstwy (Trenowanie przez jedną iterację)
        """
        path = self._select(node)
        leaf = path[-1]  # This is named leaf, because it will be the leaf in the tree of visited nodes, but not necessarily the leaf in the tree of the game
        self._expand(leaf)
        leaf_copy = make_Node_from_Position(leaf)  # Copy, so that the simulated will not overwrite the expanded board
        result = self._simulate(leaf_copy)
        self._backpropagate(path, result)

    def _select(self, node) -> list:
        """
        Znajduje niezbadanego potomka `węzła`
        """
        path = []
        while True:
            path.append(node)
            if node not in self.children or not self.children[node]:  # node is either unexplored or leaf
                return path
            unexplored = self.children[node] - self.children.keys()
            if unexplored:
                n = unexplored.pop()
                path.append(n)
                return path
            if self.selection_type == "UCT":
                node = self._uct_selection(node)  # descend a layer deeper
            elif self.selection_type == "PUCT":
                node = self._puct_selection(node)
            else:
                raise Exception("Wrong selection_type value are chosen!")

    def _uct_selection(self, node) -> Node:
        """
        Wybiera potomka węzła za pomocą metody selekcji UCT bez modyfikacji
        """
        assert all(n in self.children for n in self.children[node])  # all children of node should already be expanded
        log_N_vertex = math.log(self.N[node])

        def uct(a):
            """
            Upper confidence bound for trees (UTC)
            """
            return self.Q[a] / self.N[a] + self.C * math.sqrt(log_N_vertex / self.N[a])

        return max(self.children[node], key=uct)

    def _puct_selection(self, node) -> Node:
        """
        Wybiera potomka węzła za pomocą metody selekcji PUCT
        """
        assert all(n in self.children for n in self.children[node])  # all children of node should already be expanded
        if len(node.find_children()) <= 1:
            return node.find_children()

        N_vertex = self.N[node]
        log_N_vertex = math.log(self.N[node])
        sqrt_N_vertex = math.sqrt(log_N_vertex / self.N[node])
        M = node.calculate_weight()

        def puct(a):
            """
            Predictor + Upper confidence bound for trees = PUTC
            """
            def m(a):
                if N_vertex > 1:
                    return 2 / M[move_int_to_str(which_move_was_made(node.board, a.board))] * sqrt_N_vertex
                else:
                    return 2 / M[move_int_to_str(which_move_was_made(node.board, a.board))]
            out = self.Q[a] / self.N[a] + self.C * math.sqrt(log_N_vertex / self.N[a]) - m(a)
            return out

        return max(self.children[node], key=puct)

    def _expand(self, node) -> None:
        """
        Aktualizuje słownik `children` o dzieci `węzła`
        """
        if node in self.children:
            return  # already expanded
        self.children[node] = node.find_children()

    def _simulate(self, node):
        """
        Zwraca wynik losowej symulacji z `węzła`

        Zgodnie z oczekiwaniami, znalezienie liścia z losowymi ruchami w grze Taifho zajmuje ponad 10 000 iteracji.
        To nie jest praktyczna metoda.
        """
        invert_result = True
        while True:
            if node.moves_made % 500 == 0:  # TODO(Tak to wygląda dla podstawowego MCTSa. Bez sensu jest go wywoływać)
                node.draw_board()
                print(f"{node.moves_made} random moves")
            if node.is_leaf():
                result = node.result()
                if invert_result:
                    return self._invert_result(result)
                else:
                    return result
            node = node.find_random_child()
            invert_result = not invert_result

    def _backpropagate(self, path, result) -> None:
        """
        Wysyła wynik z powrotem do korzenia drzewa
        """
        for node in reversed(path):
            self.N[node] += 1
            self.Q[node] += result
            result = self._invert_result(result)

    def _invert_result(self, result):
        """
        Wynik gry to 0 lub 1.
        Z perspektywy przeciwnika wynik to 1 - wynik
        """
        return 1 - result


class MCTS_with_heuristic_h(MCTS):

    def __init__(self, C=math.sqrt(2), selection_type="UCT", steps=5):
        super().__init__(C, selection_type)
        self.steps = int(steps)

    def _invert_result(self, result):
        """
        Wynikiem gry jest liczba rzeczywista z heurystyki.
        Z perspektywy przeciwnika wynik jest - wynik
        """
        return - result

    def _simulate(self, node):
        """
        Zwraca wynik losowej symulacji z `węzła`
        """
        invert_result = True
        for i in range(self.steps):  # this line is changed from super()._simulate()
            if node.is_leaf():
                break  # this line is changed from super()._simulate() and code is moved out of the `for` loop
            node = node.find_random_child()
            invert_result = not invert_result
        result = node.h()  # this line is changed from super()._simulate()
        if invert_result:
            return self._invert_result(result)
        else:
            return result

class MCTS_with_heuristic_h_G(MCTS_with_heuristic_h):

    def __init__(self, C=math.sqrt(2), selection_type="UCT", steps=5, G=2):
        super().__init__(C, selection_type, steps)
        self.G = G

    def _simulate(self, node):
        """
        Zwraca wynik losowej symulacji z `węzła`
        """
        invert_result = True
        for i in range(self.steps):
            if node.is_leaf():
                break  # this line is changed from super()._simulate() and code is moved out of the `for` loop
            node = node.find_random_child()
            invert_result = not invert_result
        result = node.h_G(self.G)  # this line is changed from super()._simulate()
        if invert_result:
            return self._invert_result(result)
        else:
            return result
