import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt


def pos_from_coordinates(coordinates):
    pos = {}
    for i in range(len(coordinates)):
        pos[i] = coordinates[i]
    return pos


def plot_solution(antColony, labels=False, ax=None):
    M = antColony.problem_size
    P = [None] * M
    P[0] = "blue"
    color = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(M)]

    i = 0
    for node in antColony.best_path:
        if node == 0:
            i += 1
        else:
            P[node] = color[i - 1]

    G = nx.Graph()
    G.add_edges_from([(i, j) for i in range(M) for j in range(M) if i != j])

    labeldict = {}
    if labels:
        for node in np.arange(M):
            labeldict[node] = str(int(antColony.request[node]))

    pos = pos_from_coordinates(antColony.coordinates)
    if ax is not None:
        fig1 = nx.draw_networkx(G, pos=pos, node_color=P, labels=labeldict,
                                edgelist=[(antColony.best_path[i], antColony.best_path[i + 1]) for i in
                                          range(len(antColony.best_path) - 1)],
                                with_labels=True, ax=ax)
        ax.set_title("{}, cost = {:.0f}, trucks = {}".format(antColony.name, antColony.best_cost, antColony.best_number_of_cycles))
    else:
        fig1 = nx.draw_networkx(G, pos=pos, node_color=P, labels=labeldict,
                                edgelist=[(antColony.best_path[i], antColony.best_path[i + 1]) for i in
                                          range(len(antColony.best_path) - 1)],
                                with_labels=True)
        print(
            "Optimization with {}: iterations = {}, cost = {:.3f}, trucks = {}".format(antColony.name, antColony.iters_done,
                                                                                       antColony.best_cost,
                                                                                       antColony.best_number_of_cycles))

    return fig1


def plot_4_solutions(antColony1, antColony2, antColony3, antColony4,
                     labels=False, figsize=(15, 10), save_file=None):
    fig, axs = plt.subplots(2, 2, figsize=figsize)

    plot_solution(antColony1, labels, axs[0, 0])

    plot_solution(antColony2, labels, axs[0, 1])

    plot_solution(antColony3, labels, axs[1, 0])

    plot_solution(antColony4, labels, axs[1, 1])

    if save_file is not None:
        fig.savefig(save_file)
