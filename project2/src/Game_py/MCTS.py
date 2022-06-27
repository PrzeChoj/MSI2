import math

from .Node import Node

class MCTS:
    def __init__(self, C = math.sqrt(2), selection_type = "UCT"):
        self.C = C
        self.Q = dict(int) # total reward of each node
        self.N = dict(int) # number of visits for each node
        self.children = dict() # children for each node
        self.selection_type = selection_type

    def choose_move(self, node) -> Node:
        "Choose the best successor of node -> choose a move in the game"
        if node.is_leaf():
            raise RuntimeError(f"choose called on leaf node {node}. No more moves available!")

        if node not in self.children:
            return node.find_random_child()

        def score(a):
            if self.N[a] == 0:
                return float("-inf")  # avoid unseen moves
            return self.Q[a] / self.N[a]  # average reward

        return max(self.children[node], key=score)

    def do_rollout(self, node) -> None:
        "Make the tree one layer better. (Train for one iteration.)"
        path = self._select(node)
        leaf = path[-1]
        self._expand(leaf)
        result = self._simulate(leaf)
        self._backpropagate(path, result)

    def _select(self, node) -> list:
        "Find an unexplored descendent of `node`"
        path = []
        while True:
            path.append(node)
            if node not in self.children or not self.children[node]: # node is either unexplored or leaf
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
        assert all(n in self.children for n in self.children[node]) # all children of node should already be expanded
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
        assert all(n in self.children for n in self.children[node]) # all children of node should already be expanded
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
                    2 / M[a] * math.sqrt( math.log(N) / N )
                else:
                    return 2 / M[a]
            return self.Q[a] / self.N[a] + self.C * math.sqrt(log_N_vertex / self.N[a]) - m(self.N[a], a)

        return max(self.children[node], key=puct)

    def _expand(self, node) -> None:
        "Update the `children` dict with the children of `node`"
        if node in self.children:
            return  # already expanded
        self.children[node] = node.find_children()

    def _simulate(self, node):
        "Returns the result for a random simulation of `node`"
        invert_result = True
        while True:
            if node.is_leaf():
                result = node.result()
                if invert_result:
                    return 1 - result
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
            result = 1 - result
