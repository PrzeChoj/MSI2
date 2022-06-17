import math

from .Node import Node

class MCTS:
    def __init__(self, C = math.sqrt(2)):
        self.C = C
        self.Q = dict(int) # total reward of each node
        self.N = dict(int) # number of visits for each node
        self.children = dict() # children for each node

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
            node = self._uct_selection(node)  # descend a layer deeper

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

    def _expand(self, node) -> None:
        "Update the `children` dict with the children of `node`"
        if node in self.children:
            return  # already expanded
        self.children[node] = node.find_children()

    def _simulate(self, node):  # ToDo - napisać co zwraca
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
