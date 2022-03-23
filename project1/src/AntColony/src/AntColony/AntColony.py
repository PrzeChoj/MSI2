import numpy as np
import math
from abc import abstractmethod
from collections import Counter

from sklearn.cluster import KMeans as kmeans


class Greedy:
    def __init__(self, print_progress=False):
        self.print_progress = print_progress
        self.iters_done = None

    def set_problem(self, coordinates, request, capacity, s_max, number_of_cars):
        if coordinates.shape[1] != 2:
            raise Exception("coordinates are not in IR^2")
        if request[0] != 0:
            print("Adding 0 as technical request for warehouse")
            request = np.insert(request, 0, 0)
        if coordinates.shape[0] != request.shape[0]:
            raise Exception("Different number of coordinates and requests")
        if np.sum(request <= 0) > 1:
            raise Exception("There is non positive request, other than the technical one at the warehouse")
        if np.any(request > capacity):
            raise Exception("There is request bigger than capacity. This problem is unsolvable!")
        if number_of_cars < 0:
            raise Exception("number_of_cars parameter has to be positive")

        self.problem_size = coordinates.shape[0]
        self.coordinates = coordinates
        self.request = request
        self.capacity = capacity

        self.calculate_distance_matrix()

        if np.any(self.distance_matrix[0] * 2 > s_max):
            raise Exception(
                "There is distance bigger than max_s / 2. This problem is unsolvable! Check antColony.distance_matrix")

        self.s_max = s_max

        self.restart()

    def restart(self):
        self.best_cost = np.inf
        self.best_path = [0]
        self.best_number_of_cycles = 0

    def calculate_distance_matrix(self):
        self.distance_matrix = np.zeros((self.problem_size, self.problem_size))

        for i in range(0, self.problem_size):
            for j in range(i + 1, self.problem_size):  # only above the diagonal
                d = np.linalg.norm(self.coordinates[i, :] - self.coordinates[j, :])
                self.distance_matrix[i, j] = d
                # in this loop only the places above diagonal are filled

        self.distance_matrix += self.distance_matrix.T  # fill below diagonal

        # check if there is a 0 - distance pair
        modified_distance_matrix = self.distance_matrix + np.eye(self.problem_size)
        if modified_distance_matrix.min() == 0.0:
            if self.print_progress:
                print("There is a pair of destinations with the same coordinates:")
            for i in range(self.problem_size):
                for j in range(self.problem_size):
                    if np.all(self.coordinates[i] == self.coordinates[j]) and i != j:
                        self.distance_matrix[i, j] += 0.001
                        if self.print_progress:
                            print("{}, {}".format(i, j))

        # matrix of the distance to warehouse throughout other node
        self.distance_matrix_to_warehouse = self.distance_matrix + self.distance_matrix[0]

    def optimize(self):
        capacity_left = self.capacity
        dist_from_warehouse = 0
        node = self.best_path[-1]
        while len(self.best_path) - self.best_number_of_cycles < self.problem_size:
            possible_nodes = self.get_possible_nodes(self.best_path, capacity_left, dist_from_warehouse)
            if np.any(possible_nodes):
                next_node = np.arange(self.problem_size)[possible_nodes][self.distance_matrix[node, possible_nodes].argmin()]
            else:
                raise Exception("There is nowhere to go. This should never happen")
            self.best_path.append(next_node)
            nextnode = self.best_path[-1]
            dist_from_warehouse += self.distance_matrix[node, nextnode]
            if nextnode == 0:
                self.best_number_of_cycles += 1
                capacity_left = self.capacity
                dist_from_warehouse = 0
            else:
                capacity_left -= self.request[nextnode]

            node = nextnode

        self.best_path.append(0)
        self.best_number_of_cycles += 1
        self.best_cost = self.calculate_cost(self.best_path)


    def get_possible_nodes(self, visited_nodes, capacity_left, dist_from_warehouse):
        node = visited_nodes[-1]

        possible_nodes = np.ones(self.problem_size).astype(bool)  # (True, True, ..., True)

        possible_nodes[visited_nodes] = False

        possible_nodes[capacity_left < self.request] = False

        possible_nodes[dist_from_warehouse + self.distance_matrix_to_warehouse[node] > self.s_max] = False

        if not np.any(np.array(possible_nodes)) and node != 0:
            possible_nodes[0] = True  # If I cannot go anywhere, I'll go to the warehouse

        return possible_nodes

    def calculate_cost(self, path):
        cost_sum = 0
        for i in range(len(path) - 1):
            cost_sum += self.distance_matrix[path[i], path[i + 1]]
        return cost_sum



class AntColony:
    class Solution:
        def __init__(self, path, cost, trucks):
            self.path = path
            self.cost = cost
            self.trucks = trucks

    def __init__(self, number_of_ants=None, alpha=1, beta=1,
                 starting_pheromone=1, Q=1, ro=0.9, print_progress=False):
        if number_of_ants is not None and number_of_ants <= 0:
            raise Exception("Negative number of ants")
        if number_of_ants is not None and number_of_ants != math.floor(number_of_ants):
            raise Exception("Number of ants is not an intefer")
        if alpha < 0:
            raise Exception("Alpha parameter has to be positive")
        if beta < 0:
            raise Exception("Beta parameter has to be positive")
        if starting_pheromone < 0:
            raise Exception("starting_pheromone parameter has to be positive")
        if Q < 0:
            raise Exception("Q parameter has to be positive")
        if ro < 0 or ro > 1:
            raise Exception("ro parameter has to be in [0,1] range")

        self.number_of_ants = number_of_ants
        self.alpha = alpha
        self.beta = beta
        self.starting_pheromone = starting_pheromone
        self.Q = Q
        self.ro = ro
        self.print_progress = print_progress

    def set_problem(self, coordinates, request, capacity, s_max, number_of_cars):
        if coordinates.shape[1] != 2:
            raise Exception("coordinates are not in IR^2")
        if request[0] != 0:
            print("Adding 0 as technical request for warehouse")
            request = np.insert(request, 0, 0)
        if coordinates.shape[0] != request.shape[0]:
            raise Exception("Different number of coordinates and requests")
        if np.sum(request <= 0) > 1:
            raise Exception("There is non positive request, other than the technical one at the warehouse")
        if np.any(request > capacity):
            raise Exception("There is request bigger than capacity. This problem is unsolvable!")
        if number_of_cars < 0:
            raise Exception("number_of_cars parameter has to be positive")

        self.problem_size = coordinates.shape[0]
        self.coordinates = coordinates
        self.request = request
        self.capacity = capacity

        self.restart()

        if self.number_of_ants is None:  # other numbers of ants is set in __init__
            self.number_of_ants = 2 * self.problem_size

        if np.any(self.distance_matrix[0] * 2 > s_max):
            raise Exception(
                "There is distance bigger than max_s / 2. This problem is unsolvable! Check antColony.distance_matrix")

        self.s_max = s_max
        self.number_of_cars = number_of_cars

        self.best_solutions = []

    def restart(self):  # IF YOU MODIFY THIS FUNCTION, ALSO MODIFY THE ONE IN AntColony_Abstract_Modification
        self.pheromone_restart()
        self.calculate_distance_matrix()
        self.calculate_transition_matrix()

        self.best_cost = np.inf
        self.best_path = None
        self.best_number_of_cycles = None
        self.now_iter = 0
        self.iters_done = 0

    def calculate_distance_matrix(self):
        self.distance_matrix = np.zeros((self.problem_size, self.problem_size))

        for i in range(0, self.problem_size):
            for j in range(i + 1, self.problem_size):  # only above the diagonal
                d = np.linalg.norm(self.coordinates[i, :] - self.coordinates[j, :])
                self.distance_matrix[i, j] = d
                # in this loop only the places above diagonal are filled

        self.distance_matrix += self.distance_matrix.T  # fill below diagonal

        # check if there is a 0 - distance pair
        modified_distance_matrix = self.distance_matrix + np.eye(self.problem_size)
        if modified_distance_matrix.min() == 0.0:
            if self.print_progress:
                print("There is a pair of destinations with the same coordinates:")
            for i in range(self.problem_size):
                for j in range(self.problem_size):
                    if np.all(self.coordinates[i] == self.coordinates[j]) and i != j:
                        self.distance_matrix[i, j] += 0.001
                        if self.print_progress:
                            print("{}, {}".format(i, j))

        # matrix of the distance to warehouse throughout other node
        self.distance_matrix_to_warehouse = self.distance_matrix + self.distance_matrix[0]

    def calculate_transition_matrix(self):
        # self.T_P is non standardize Probability
        modified_distance_matrix = self.distance_matrix + np.eye(self.problem_size)
        self.T_P = (self.pheromone_matrix ** self.alpha) * ((1 / modified_distance_matrix) ** self.beta)

        self.T_P = (self.T_P.T / self.T_P.sum(axis=1)).T  # T_P is matrix of probabilities

        self.T_P = self.T_P + 0.01 * np.ones_like(self.T_P)  # modification for not relying only on pheromone

        np.fill_diagonal(self.T_P, 0)

    def pheromone_restart(self):
        self.pheromone_matrix = self.starting_pheromone * (np.ones((self.problem_size,
                                                                    self.problem_size)) - np.eye(self.problem_size))

    def ant_find_path(
            self):  # return (list of nodes, number of times to start from warehouse - number of 0 in list of nodes minus 1 - number of ant_find_circle() calls)
        visited_nodes = [0]  # start from warehouse
        number_of_cycles = 0
        while len(
                visited_nodes) - number_of_cycles < self.problem_size:  # in visited_nodes should be [0,1,...,self.problem_size] and additionally number_of_cycles zeros
            number_of_cycles += 1
            visited_nodes = self.ant_find_circle(visited_nodes)

        return visited_nodes, number_of_cycles

    def ant_find_circle(self, visited_nodes):
        if visited_nodes is None or len(visited_nodes) == 0 or visited_nodes[-1] != 0:
            raise Exception("ant is not at the warehouse")

        node = 0
        dist_from_warehouse = 0
        capacity_left = self.capacity

        # do_while loop
        move = self.ant_make_move(visited_nodes, capacity_left, dist_from_warehouse)
        visited_nodes.append(move)
        dist_from_warehouse += self.distance_matrix[node, move]
        capacity_left -= self.request[move]
        node = move
        while (node != 0):
            move = self.ant_make_move(visited_nodes, capacity_left, dist_from_warehouse)
            visited_nodes.append(move)
            dist_from_warehouse += self.distance_matrix[node, move]
            capacity_left -= self.request[move]
            node = move

        return visited_nodes

    def ant_make_move(self, visited_nodes, capacity_left,
                      dist_from_warehouse):  # Draw a node from probability according to self.T_P[possible_nodes]
        node = visited_nodes[-1]
        if node == 0 and dist_from_warehouse != 0:
            raise Exception("ant is in warehouse, but the dist_from_warehouse is not 0")
        if capacity_left < 0:
            raise Exception("There is less than none capacity left")

        possible_nodes = self.get_possible_nodes(visited_nodes, capacity_left, dist_from_warehouse)

        if not np.any(possible_nodes):
            raise Exception("There is nowhere to go :(")

        my_cum_sum = np.cumsum(self.T_P[node, possible_nodes])
        c = my_cum_sum[-1]
        if c == 0:
            raise Exception("The probability is 0 :(")

        u = self.uniform_drawn[self.now_iter, self.ant_now, len(visited_nodes)] * c

        return np.where(possible_nodes)[0][np.where(u < my_cum_sum)[0][0]]  # EXP: np.where(u < my_cum_sum) == ([3,4,5],) # so the 3 was drawn



    def get_possible_nodes(self, visited_nodes, capacity_left, dist_from_warehouse):
        node = visited_nodes[-1]

        possible_nodes = np.ones(self.problem_size).astype(bool)  # (True, True, ..., True)

        possible_nodes[visited_nodes] = False

        possible_nodes[0] = True  # If we are in 0, then still self.T_P [0,0] = 0, so this can stay True

        possible_nodes[capacity_left < self.request] = False

        possible_nodes[dist_from_warehouse + self.distance_matrix_to_warehouse[node] > self.s_max] = False

        return possible_nodes

    def calculate_cost(self, path):
        cost_sum = 0
        for i in range(len(path) - 1):
            cost_sum += self.distance_matrix[path[i], path[i + 1]]
        return cost_sum

    def pheromone_modify(self, paths, costs):
        self.pheromone_matrix *= (1 - self.ro)
        delta_pheromone_matrix = np.zeros_like(self.pheromone_matrix)
        for path in paths:
            # e.g. path = [0,4,3,0,1,5,2,0]
            for i in range(len(path) - 1):
                delta_pheromone_matrix[path[i], path[i + 1]] += 1 / costs[i]
        self.pheromone_matrix += self.Q * delta_pheromone_matrix

    def single_iteration(self, iteration):
        paths = [None for i in range(self.number_of_ants)]
        num_of_cycles = [None for i in range(self.number_of_ants)]
        for i in range(self.number_of_ants):
            self.ant_now = i
            paths[i], num_of_cycles[i] = self.ant_find_path()
        costs = np.array([self.calculate_cost(paths[i]) for i in range(self.number_of_ants)])

        i_min_cost = np.argmin(costs)
        if costs[i_min_cost] < self.best_cost:
            self.best_solutions.append(self.Solution(paths[i_min_cost], costs[i_min_cost], num_of_cycles[i_min_cost]))

            self.best_path = paths[i_min_cost]
            self.best_cost = costs[i_min_cost]
            self.best_number_of_cycles = num_of_cycles[i_min_cost]
            if self.print_progress:
                print("New best solution in {} iteration: cost = {:.3f} and uses {} trucks".format(iteration,
                                                                                                   self.best_cost,
                                                                                                   self.best_number_of_cycles))

        self.pheromone_modify(paths, costs)
        self.calculate_transition_matrix()

    def optimize(self, max_iter, print_progress=False, rng_seed=None, restart=True, check_cars=True):
        self.print_progress = print_progress
        self.rng = np.random.default_rng(rng_seed)
        if restart:
            self.restart()

        self.uniform_drawn = self.rng.uniform(size=(max_iter, self.number_of_ants, 2 * self.problem_size))

        self.now_iter = 0
        for i in range(max_iter):
            self.iters_done += 1
            self.single_iteration(i)
            self.now_iter += 1

        if check_cars:
            # TODO - Czy udało się znaleść rozwiązanie z dobrą liczbą samochodów?
            pass


class AntColony_Abstract_Modification(AntColony):
    def restart(self):
        self.pheromone_restart()
        self.calculate_distance_matrix()
        self.calculate_legal_edges()  # this is added in modified version of restart() function
        self.calculate_transition_matrix()

        self.best_cost = np.inf
        self.best_path = None
        self.best_cycles_number = None

        self.now_iter = 0
        self.iters_done = 0

    @abstractmethod
    def calculate_legal_edges(self):
        pass

    def calculate_transition_matrix(self):
        super(AntColony_Abstract_Modification, self).calculate_transition_matrix()

        self.T_P *= self.legal_edges


class AntColony_Reduced(AntColony_Abstract_Modification):
    def calculate_legal_edges(self):
        number_of_neighbours = math.floor(math.sqrt(self.problem_size)) + 1

        argsorted_distance_matrix = np.argsort(self.distance_matrix)[:, number_of_neighbours:]

        self.legal_edges = np.ones_like(self.distance_matrix, dtype=bool)

        for i in range(self.problem_size):
            self.legal_edges[i, argsorted_distance_matrix[i]] = False

        # one can always go to and from the warehouse
        self.legal_edges[:, 0] = True
        self.legal_edges[0, :] = True
        self.legal_edges[0, 0] = False


class AntColony_Divided_Cluster(AntColony_Abstract_Modification):
    # This is just the class for optimizing the single cluster of Divided modification. The interface for using the Divided modification is in AntColony_Divided class.
    # This class will get all information about all the nodes, but will only work for the subset of them

    def set_problem(self, coordinates, request, capacity, s_max, number_of_cars,
                    subset_of_nodes_to_solve):
        if len(subset_of_nodes_to_solve) >= coordinates.shape[0]:
            raise Exception("There is too much nodes to be solved")
        if np.any(subset_of_nodes_to_solve == 0):
            raise Exception("There is the warehouse in subsets to optimize")

        self.subset_of_nodes_to_solve = subset_of_nodes_to_solve
        self.problem_size_divided = len(subset_of_nodes_to_solve)
        super(AntColony_Divided_Cluster, self).set_problem(coordinates, request, capacity, s_max, number_of_cars)

    def calculate_legal_edges(self):
        self.legal_edges = np.zeros((self.problem_size, self.problem_size), dtype=bool)

        # it is not in np, so it is not efficient, but it is only run one time per restart, not in a loop
        for i in self.subset_of_nodes_to_solve:
            for j in self.subset_of_nodes_to_solve:
                self.legal_edges[i, j] = True

        # one can always go to and from the warehouse
        self.legal_edges[0, self.subset_of_nodes_to_solve] = True
        self.legal_edges[self.subset_of_nodes_to_solve, 0] = True
        self.legal_edges[0, 0] = False

    def ant_find_path(self):
        visited_nodes = [0]
        number_of_cycles = 0
        while len(
                visited_nodes) - number_of_cycles < self.problem_size_divided + 1:  # the problem_size changed to problem_size_divided from the basic version of algorithm; +1, coz the warehouse
            number_of_cycles += 1
            visited_nodes = self.ant_find_circle(visited_nodes)

        return visited_nodes, number_of_cycles


class AntColony_Divided(AntColony):  # The same interface as AntColony, but those are only workarounds
    def set_problem(self, coordinates, request, capacity, s_max, number_of_cars):
        if coordinates.shape[1] != 2:
            raise Exception("coordinates are not in IR^2")
        if request[0] != 0:
            print("Adding 0 as technical request for warehouse")
            request = np.insert(request, 0, 0)
        if coordinates.shape[0] != request.shape[0]:
            raise Exception("Different number of coordinates and requests")
        if np.sum(request <= 0) > 1:
            raise Exception("There is non positive request, other than the technical one at the warehouse")
        if np.any(request > capacity):
            raise Exception("There is request bigger than capacity. This problem is unsolvable!")
        if number_of_cars < 0:
            raise Exception("number_of_cars parameter has to be positive")

        self.problem_size = coordinates.shape[0]
        self.coordinates = coordinates
        self.request = request
        self.capacity = capacity

        # No restart()

        # no ants

        self.calculate_distance_matrix()

        if np.any(self.distance_matrix[0] * 2 > s_max):
            raise Exception(
                "There is distance bigger than max_s / 2. This problem is unsolvable! Check antColony.distance_matrix")

        self.s_max = s_max
        self.number_of_cars = number_of_cars

        self.restart()

    def restart(self):
        self.best_solutions = []

        self.calculate_kmeans_model()

        self.best_cost = np.inf
        self.best_path = np.array([])
        self.best_path = self.best_path.astype(int)
        self.best_number_of_cycles = None

        self.best_path_clusters = [None for i in range(self.number_of_clusters)]
        self.best_cost_clusters = [None for i in range(self.number_of_clusters)]
        self.best_number_of_cycles_clusters = [None for i in range(self.number_of_clusters)]

        self.iters_done = 0

    def calculate_kmeans_model(self):
        coordinates_no_warehouse = self.coordinates[1:]  # the warehouse will not be considered for clusters

        self.number_of_clusters = math.floor(math.log(len(coordinates_no_warehouse)))
        self.kmeans_model = kmeans(n_clusters=self.number_of_clusters,
                                   random_state=0).fit(coordinates_no_warehouse)

        self.clusters = [np.where(self.kmeans_model.labels_ == i)[0] + 1  # +1, coz the warehouse is not in kmeans model
                         for i in range(self.number_of_clusters)]

        if self.print_progress:
            counter = Counter(self.kmeans_model.labels_)
            counts = [x[1] for x in counter.items()]
            print("{} clusters used with counts {}".format(self.number_of_clusters, counts))

    def cluster_solve(self, i, max_iter, rng_seed, check_cars):
        cluster_solver = AntColony_Divided_Cluster(self.number_of_ants, self.alpha, self.beta, self.starting_pheromone,
                                                   self.Q, self.ro, self.print_progress)
        cluster_solver.set_problem(self.coordinates, self.request, self.capacity, self.s_max, self.number_of_cars,
                                   self.clusters[i])

        cluster_solver.optimize(max_iter, print_progress=self.print_progress, restart=True, rng_seed=rng_seed,
                                check_cars=check_cars)

        self.best_path_clusters[i] = cluster_solver.best_path
        self.best_cost_clusters[i] = cluster_solver.best_cost
        self.best_number_of_cycles_clusters[i] = cluster_solver.best_number_of_cycles

    def optimize(self, max_iter, print_progress=False, rng_seed=None, restart=True, check_cars=True):
        self.print_progress = print_progress
        self.rng = np.random.default_rng(rng_seed)
        if restart:
            self.restart()

        for i in range(self.number_of_clusters):
            if print_progress:
                print("\nOptimization of {}th cluster:".format(i))
            self.cluster_solve(i, max_iter, rng_seed, check_cars)

        self.best_cost = sum(self.best_cost_clusters)
        self.best_number_of_cycles = sum(self.best_number_of_cycles_clusters)

        for i in range(self.number_of_clusters):
            path = np.array(self.best_path_clusters[i][0:-1])
            path = path.astype(int)
            self.best_path = np.concatenate((self.best_path, np.array(path)))

        self.best_path = np.append(self.best_path, 0)

        self.iters_done += max_iter

        if check_cars:
            # TODO - Czy udało się znaleść rozwiązanie z dobrą liczbą samochodów?
            pass
