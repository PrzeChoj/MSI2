import math
from collections import defaultdict

from Node import Node, make_Node_from_Position


class MCTS:
    def __init__(self, C=math.sqrt(2), selection_type="UCT"):
        self.C = C
        self.Q = defaultdict(int)  # total reward of each node
        self.N = defaultdict(int)  # number of visits for each node
        self.children = dict()  # children for each node
        self.selection_type = selection_type

    def choose_move(self, node) -> Node:
        """Choose the best successor of node -> choose a move in the game"""
        if node.is_leaf():
            raise RuntimeError(f"choose called on leaf node. No more moves available!")

        if node not in self.children:
            return node.find_random_child()

        def score(a):
            if self.N[a] == 0:
                return float("-inf")  # avoid unseen moves
            return self.Q[a] / self.N[a]  # average reward

        return max(self.children[node], key=score)  # TODO(To jest następny node, a my chcemy int, numer ruchu)

    def do_rollout(self, node) -> None:
        """Make the tree one layer better. (Train for one iteration.)"""
        path = self._select(node)
        leaf = path[-1]  # This is named leaf, because we will be the leaf in the tree of visited nodes, but not necessarily the leaf in the tree of the game
        self._expand(leaf)  # ToDO(doszło do NoneType i wywaliło błąd) # TODO(Paula, czy nadal tak masz, bo dużo pozmieniałem i może jeuż jest ok.)
        leaf_copy = make_Node_from_Position(leaf)  # Copy, so that the simulated will not overwrite the expanded board
        result = self._simulate(leaf_copy)
        self._backpropagate(path, result)

    def _select(self, node) -> list:
        """Find an unexplored descendent of `node`"""
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
        Select a child of node using UCT selection method with no modification
        """
        assert all(n in self.children for n in self.children[node])  # all children of node should already be expanded
        log_N_vertex = math.log(self.N[node])

        def uct(a):
            """
            Upper confidence bound for trees
            """
            return self.Q[a] / self.N[a] + self.C * math.sqrt(log_N_vertex / self.N[a])

        return max(self.children[node], key=uct)

    def _puct_selection(self, node) -> Node:
        """
        Select a child of node using UCT selection method
        """
        assert all(n in self.children for n in self.children[node])  # all children of node should already be expanded
        if len(node.find_children()) <= 1:
            return node.find_children()

        log_N_vertex = math.log(self.N[node])
        M = node.calculate_weight()

        def puct(a):
            """
            Predictor + Upper confidence bound for trees
            """
            def m(N, a):
                if N > 1:
                    2 / M[a] * math.sqrt(math.log(N) / N)
                else:
                    return 2 / M[a]
            return self.Q[a] / self.N[a] + self.C * math.sqrt(log_N_vertex / self.N[a]) - m(self.N[a], a)

        return max(self.children[node], key=puct)

    def _expand(self, node) -> None:
        """Update the `children` dict with the children of `node`"""
        if node in self.children:
            return  # already expanded
        self.children[node] = node.find_children()

    def _simulate(self, node):
        """
        Returns the result for a random simulation from `node`

        As expected, it takes over 10 000 iterations to find a leaf with random moves in Taifho game.
        It is not practical method.
        """
        invert_result = True
        while True:
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
        Send the result back up to the root
        """
        for node in reversed(path):
            self.N[node] += 1
            self.Q[node] += result
            result = self._invert_result(result)

    def _invert_result(self, result):
        """
        The result of the game is 0 or 1.
        From the opponent perspective, the result is 1 - result
        """
        return 1 - result


class MCTS_with_heuristic_h(MCTS):

    def __init__(self, C=math.sqrt(2), selection_type="UCT", depth=5):
        super().__init__(C, selection_type)
        self.depth = depth

    def _invert_result(self, result):
        """
        The result of the game is real number from the heuristic.
        From the opponent perspective, the result is - result
        """
        return - result

    def _simulate(self, node):
        """Returns the result for a random simulation of `node`"""
        invert_result = True
        for i in range(self.depth):  # this line is changed from super()._simulate()
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

    def __init__(self, C=math.sqrt(2), selection_type="UCT", depth=5, G=2):
        super().__init__(C, selection_type, depth)
        self.G = G

    def _simulate(self, node):
        """Returns the result for a random simulation of `node`"""
        invert_result = True
        for i in range(self.depth):
            if node.is_leaf():
                break  # this line is changed from super()._simulate() and code is moved out of the `for` loop
            node = node.find_random_child()
            invert_result = not invert_result
        result = node.h_G(self.G)  # this line is changed from super()._simulate()
        if invert_result:
            return self._invert_result(result)
        else:
            return result
