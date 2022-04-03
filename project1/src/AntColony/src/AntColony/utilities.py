import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt
import ast as _ast
import logging
import re
import os
from sys import maxsize
from .AntColony import *
from .Problem import Problem

rng = np.random.default_rng()

np.set_printoptions(suppress=True,  # no scientific notation
                    threshold=maxsize,  # logging the whole np.arrays
                    linewidth=np.inf)  # one line for vectors

# Font for matplotlib plots:
font = {'weight' : 'bold',
        'size'   : 16}
plt.rc('font', **font)


def pos_from_coordinates(coordinates):
    pos = {}
    for i in range(len(coordinates)):
        pos[i] = coordinates[i]
    return pos


def plot_solution(antColony, labels=False, ax=None, start_title=""):
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
        ax.set_title(start_title + "{}\ncost = {:.0f}, trucks = {}, iters = {}".format(antColony.name,
                                                                         antColony.best_cost,
                                                                         antColony.best_number_of_cycles,
                                                                         antColony.iters_done))
    else:
        fig1 = nx.draw_networkx(G, pos=pos, node_color=P, labels=labeldict,
                                edgelist=[(antColony.best_path[i], antColony.best_path[i + 1]) for i in
                                          range(len(antColony.best_path) - 1)],
                                with_labels=True)
        print(
            "Optimization with {}: iterations = {}, cost = {:.3f}, trucks = {}".format(antColony.name,
                                                                                       antColony.iters_done,
                                                                                       antColony.best_cost,
                                                                                       antColony.best_number_of_cycles))

    return fig1


def plot_4_solutions(antColony1, antColony2, antColony3, antColony4,
                     labels=False, figsize=(15, 10), save_file=None, start_title=""):
    fig, axs = plt.subplots(2, 2, figsize=figsize)

    plot_solution(antColony1, labels, axs[0, 0], start_title)

    plot_solution(antColony2, labels, axs[0, 1], start_title)

    plot_solution(antColony3, labels, axs[1, 0], start_title)

    plot_solution(antColony4, labels, axs[1, 1], start_title)

    if save_file is not None:
        fig.savefig(save_file)


def save_to_file(antColont, file):
    # we assume the "logs" folder already exists
    try:
        pattern = re.compile("\/[^\/]*$")  # find the moment where the folder name ends and file name starts
        pattern_search = pattern.search(file)
        os.mkdir(file[0:pattern_search.start()])
    except:
        pass  # folder already exists

    try:
        with open(file, 'w') as myfile:
            myfile.write("best_cost = {}".format(antColont.best_cost))
            myfile.write("\nbest_number_of_cycles = {}".format(antColont.best_number_of_cycles))
            myfile.write("\nbest_path = {}".format(antColont.best_path))
            myfile.write("\niters_done = {}".format(antColont.iters_done))
            myfile.write("\n")

            if antColont.name != "Greedy":
                myfile.write("\nalpha = {}".format(antColont.alpha))
                myfile.write("\nbeta = {}".format(antColont.beta))
                myfile.write("\ncars_penalty = {}".format(antColont.cars_penalty))
                myfile.write("\nnumber_of_ants = {}".format(antColont.number_of_ants))
                myfile.write("\nnumber_of_cars = {}".format(antColont.number_of_cars))
                myfile.write("\nQ = {}".format(antColont.Q))
                myfile.write("\nro = {}".format(antColont.ro))
                myfile.write("\ntime_of_optimization = {}".format(antColont.time_of_optimization))
                myfile.write("\n")
    except:
        print('Something is wrong. Most probably you dont have the "logs" folder. Create it manually.')


def args_to_dict(args, known_names, specials=None, split='=',  # copied from cocoex; usefun in script.py
                 print=lambda *args, **kwargs: None):
    def eval_value(value):
        try:
            return _ast.literal_eval(value)
        except ValueError:
            return value

    res = {}
    for arg in args:
        name, value = arg.split(split)
        # what remains to be done is to verify name,
        # compute non-string value, and assign res[name] = value
        if specials and name in specials:
            if name == 'batch':
                print('batch:')
                for k, v in zip(specials['batch'].split('/'), value.split('/')):
                    res[k] = int(v)  # batch accepts only int
                    print(' ', k, '=', res[k])
                continue  # name is processed
            else:
                raise ValueError(name, 'is unknown special')
        for known_name in known_names if known_names is not None else [name]:
            # check that name is an abbreviation of known_name and unique in known_names
            if known_name.startswith(name) and (
                    sum([other.startswith(name)
                         for other in known_names or [problem_names]]) == 1):
                res[known_name] = eval_value(value)
                print(known_name, '=', res[known_name])
                break  # name == arg.split()[0] is processed
        else:
            raise ValueError(name, 'not found or ambiguous in `known_names`')
    return res


# logs for Divided were missing the "," sing. This should fix it
def fix_text(text):
    return re.sub("[0-9]\s+", lambda number_and_space: number_and_space[0] + ", ", text)


def plot_4_medians(problem_id, labels=False, is_save_file=False):
    problem_id -= 1
    problem_strings = ["./../../data/augerat/A-n32-k05.xml",
                       "./../../data/augerat/A-n44-k06.xml",
                       "./../../data/augerat/A-n60-k09.xml",
                       "./../../data/augerat/A-n69-k09.xml",
                       "./../../data/uchoa/X-n101-k25.xml",
                       "./../../data/uchoa/X-n120-k6.xml"]

    photo_files = ["graphs/median_1.png", "graphs/median_2.png", "graphs/median_3.png",
                   "graphs/median_4.png", "graphs/median_5.png", "graphs/median_6.png"]

    problem_names = ["Small 1", "Small 2", "Medium 1", "Medium 2", "Big 1", "Big 2"]

    problems = [Problem(file) for file in problem_strings]
    s_maxs = [205, 285, 223, 223, 1750, 2641.2]
    max_cars = [7, 6, 9, 10, 28, 8]

    # data from logs:
    greedy_number_of_cycles = [8, 7, 10, 10, 29, 10]
    basic_number_of_cycles = [7, 8, 11, 12, 29, 17]
    reduced_number_of_cycles = [9, 9, 17, 13, 33, 20]
    divided_number_of_cycles = [8, 8, 12, 13, 32, 13]

    greedy_costs = [1547.141046024138, 1412.5452434839606, 1902.126315747555, 1575.314230122054, 39789.21749651673,
                    24634.76762421654]
    basic_costs = [1231.6096926285666, 1342.5325334438785, 2124.705733918729, 1995.34554739404, 40487.79608446973,
                   42880.93573898017]
    reduced_costs = [1318.4147299144104, 1321.12260016393, 2009.8719883614842, 1752.0805856211862, 36397.165270412734,
                     35042.96102049668]
    divided_costs = [1271.5613952644676, 1170.0039649318753, 1696.743381789852, 1411.924718365075, 34493.28504182162,
                     29804.355234469833]

    basic_iters_dones = [692, 398, 197, 156, 63, 51]
    reduced_iters_dones = [617, 337, 176, 133, 56, 48]
    divided_iters_dones = [865, 400, 194, 148, 1, 53]

    greedy_paths = [
        [0, 30, 26, 16, 12, 1, 7, 14, 29, 0, 24, 27, 20, 5, 25, 10, 0, 13, 21, 31, 19, 17, 3, 6, 0, 18, 8, 28, 23, 0, 2,
         0, 15, 22, 9, 0, 4, 0, 11, 0],
        [0, 31, 8, 15, 28, 27, 19, 24, 0, 4, 34, 17, 12, 3, 6, 22, 9, 38, 21, 0, 2, 41, 14, 36, 25, 39, 0, 7, 5, 33, 10,
         11, 26, 30, 40, 0, 29, 43, 23, 20, 16, 18, 0, 13, 1, 35, 0, 37, 42, 32, 0],
        [0, 41, 18, 33, 52, 19, 59, 38, 55, 50, 0, 16, 20, 25, 46, 40, 11, 4, 21, 6, 2, 49, 0, 14, 47, 23, 34, 24, 58,
         35, 0, 3, 31, 28, 44, 1, 36, 22, 0, 15, 39, 26, 27, 17, 57, 37, 0, 7, 13, 8, 29, 56, 0, 53, 30, 10, 54, 5, 48,
         0, 43, 12, 32, 9, 0, 45, 42, 0, 51, 0],
        [0, 19, 24, 43, 52, 26, 54, 5, 46, 0, 28, 34, 14, 42, 57, 11, 63, 37, 16, 1, 10, 0, 7, 27, 65, 55, 60, 4, 47,
         25, 29, 0, 31, 18, 53, 2, 64, 61, 67, 21, 0, 58, 22, 12, 66, 23, 30, 6, 0, 62, 56, 39, 8, 59, 20, 41, 0, 50,
         48, 36, 17, 32, 40, 0, 9, 51, 3, 15, 44, 13, 35, 45, 0, 68, 38, 33, 0, 49, 0],
        [0, 32, 24, 46, 35, 15, 95, 73, 0, 50, 79, 30, 85, 64, 0, 5, 12, 58, 87, 7, 0, 21, 100, 61, 97, 34, 65, 0, 23,
         19, 11, 81, 71, 89, 0, 80, 17, 94, 0, 8, 56, 96, 26, 0, 31, 53, 57, 0, 75, 93, 33, 0, 44, 40, 3, 88, 14, 29, 0,
         22, 41, 20, 76, 0, 18, 4, 39, 72, 47, 0, 77, 67, 28, 0, 59, 60, 82, 0, 27, 91, 62, 0, 63, 25, 42, 0, 10, 78, 6,
         43, 0, 38, 48, 37, 0, 52, 83, 0, 66, 84, 90, 68, 0, 1, 86, 54, 69, 0, 51, 99, 0, 92, 9, 55, 0, 13, 74, 0, 98,
         0, 16, 70, 0, 49, 2, 0, 36, 0, 45, 0],
        [0, 20, 116, 107, 61, 119, 94, 10, 88, 52, 50, 38, 31, 73, 25, 29, 80, 16, 84, 93, 55, 70, 0, 96, 71, 54, 112,
         45, 3, 78, 39, 42, 49, 2, 43, 76, 113, 64, 109, 82, 98, 13, 99, 9, 0, 32, 91, 21, 12, 79, 118, 30, 67, 33, 23,
         14, 58, 83, 15, 44, 18, 17, 103, 8, 0, 87, 72, 62, 40, 111, 102, 34, 27, 11, 56, 1, 24, 37, 57, 100, 77, 53,
         90, 114, 0, 6, 92, 35, 59, 22, 4, 95, 60, 68, 81, 69, 85, 89, 0, 36, 115, 48, 65, 86, 74, 110, 106, 7, 97, 19,
         28, 0, 41, 5, 26, 117, 108, 66, 46, 75, 0, 51, 63, 104, 0, 47, 101, 0, 105, 0]
    ]
    basic_paths = [
        [0, 16, 12, 1, 7, 0, 30, 26, 17, 19, 31, 21, 13, 0, 28, 4, 18, 27, 0, 22, 9, 29, 5, 20, 0, 14, 24, 23, 3, 2, 6,
         0, 25, 10, 15, 0, 11, 8, 0],
        [0, 26, 10, 11, 16, 20, 18, 1, 35, 9, 0, 31, 8, 15, 28, 27, 19, 0, 25, 21, 32, 33, 7, 0, 4, 34, 39, 12, 17, 22,
         36, 38, 0, 37, 42, 5, 24, 30, 23, 0, 29, 40, 0, 14, 41, 6, 3, 43, 13, 0, 2, 0],
        [0, 56, 43, 17, 27, 13, 20, 16, 0, 1, 23, 58, 24, 3, 25, 0, 54, 10, 22, 45, 5, 2, 18, 0, 33, 41, 38, 15, 55, 35,
         34, 47, 14, 0, 11, 21, 4, 30, 44, 31, 28, 49, 46, 0, 39, 50, 9, 32, 12, 0, 40, 42, 48, 6, 0, 51, 0, 29, 7, 8,
         57, 37, 0, 26, 52, 59, 19, 0, 36, 53, 0],
        [0, 58, 22, 12, 0, 36, 1, 48, 25, 5, 46, 43, 24, 0, 19, 9, 30, 15, 3, 60, 55, 0, 52, 10, 17, 32, 41, 20, 29, 26,
         0, 45, 35, 53, 23, 31, 7, 0, 28, 42, 57, 40, 38, 50, 16, 66, 0, 37, 54, 68, 59, 8, 39, 0, 18, 4, 47, 14, 34, 0,
         49, 33, 2, 64, 61, 21, 67, 0, 27, 65, 0, 44, 13, 6, 51, 56, 62, 0, 11, 63, 0],
        [0, 53, 10, 34, 0, 23, 30, 85, 62, 71, 0, 5, 0, 19, 42, 65, 72, 57, 64, 100, 0, 49, 2, 39, 25, 0, 79, 66, 58, 0,
         73, 95, 31, 40, 14, 0, 50, 83, 56, 0, 27, 97, 80, 0, 74, 69, 3, 88, 0, 9, 16, 13, 7, 0, 43, 29, 36, 82, 0, 41,
         22, 35, 0, 67, 60, 26, 47, 87, 0, 6, 37, 44, 46, 0, 75, 93, 33, 0, 11, 61, 96, 38, 0, 32, 24, 94, 12, 0, 1, 92,
         54, 15, 0, 84, 90, 78, 28, 21, 0, 45, 63, 0, 8, 17, 76, 0, 59, 98, 89, 91, 0, 77, 48, 0, 81, 99, 51, 0, 20, 55,
         68, 0, 86, 70, 0, 18, 4, 0, 52, 0],
        [0, 20, 76, 43, 18, 17, 103, 96, 71, 80, 32, 91, 25, 29, 31, 52, 50, 0, 62, 56, 33, 118, 30, 67, 88, 38, 0, 10,
         94, 119, 45, 90, 78, 8, 79, 23, 14, 55, 93, 84, 61, 116, 107, 0, 87, 72, 53, 13, 98, 83, 15, 73, 0, 46, 66,
         108, 113, 64, 109, 42, 39, 12, 16, 0, 69, 81, 68, 35, 92, 6, 40, 34, 111, 102, 112, 0, 77, 100, 19, 28, 82, 49,
         0, 7, 97, 85, 70, 0, 117, 5, 26, 3, 114, 54, 0, 36, 115, 75, 58, 9, 99, 2, 21, 0, 4, 95, 60, 22, 51, 89, 41, 0,
         86, 65, 74, 110, 106, 44, 0, 57, 59, 37, 24, 1, 27, 11, 0, 104, 63, 0, 48, 0, 105, 0, 47, 101, 0]
    ]
    reduced_paths = [
        [0, 30, 16, 1, 12, 0, 24, 14, 26, 0, 18, 8, 11, 0, 27, 0, 20, 5, 25, 10, 15, 29, 0, 3, 23, 2, 6, 13, 0, 19, 17,
         31, 21, 7, 0, 9, 22, 0, 4, 28, 0],
        [0, 13, 36, 9, 22, 3, 12, 34, 4, 0, 18, 35, 1, 20, 30, 23, 39, 0, 26, 10, 11, 33, 24, 27, 15, 19, 0, 29, 40, 43,
         25, 17, 6, 0, 16, 42, 32, 21, 5, 28, 8, 0, 31, 0, 38, 14, 2, 41, 0, 7, 0, 37, 0],
        [0, 41, 18, 0, 14, 33, 0, 7, 37, 57, 17, 27, 26, 0, 47, 35, 55, 15, 50, 39, 58, 24, 2, 0, 11, 21, 4, 6, 28, 0,
         40, 0, 45, 22, 10, 54, 36, 49, 0, 56, 43, 12, 32, 9, 0, 53, 30, 1, 44, 31, 0, 20, 3, 0, 25, 46, 0, 59, 38, 13,
         29, 8, 0, 16, 0, 23, 34, 0, 51, 0, 42, 48, 5, 0, 19, 52, 0],
        [0, 14, 65, 47, 60, 55, 27, 34, 7, 19, 0, 11, 5, 54, 43, 52, 26, 57, 28, 0, 63, 37, 32, 17, 10, 48, 1, 0, 66,
         23, 51, 13, 35, 45, 0, 41, 20, 38, 40, 8, 39, 59, 68, 0, 33, 9, 3, 15, 6, 44, 30, 53, 18, 31, 0, 49, 61, 64, 0,
         21, 67, 2, 0, 24, 58, 12, 22, 0, 50, 16, 0, 42, 25, 4, 0, 46, 0, 56, 36, 62, 29, 0],
        [0, 52, 83, 81, 0, 99, 71, 91, 100, 0, 41, 22, 35, 0, 24, 32, 31, 95, 73, 33, 0, 84, 90, 92, 76, 69, 0, 77, 72,
         57, 82, 87, 88, 0, 3, 40, 44, 34, 64, 96, 0, 54, 9, 68, 0, 39, 10, 25, 14, 0, 30, 93, 0, 4, 18, 0, 61, 19, 0,
         97, 27, 0, 66, 74, 13, 0, 75, 85, 79, 21, 0, 1, 86, 0, 78, 37, 6, 65, 28, 0, 43, 36, 29, 49, 0, 17, 80, 56, 0,
         16, 55, 0, 51, 62, 89, 0, 11, 50, 0, 15, 20, 46, 0, 53, 0, 5, 12, 58, 0, 26, 48, 60, 0, 2, 7, 0, 23, 8, 0, 38,
         47, 94, 59, 0, 45, 42, 63, 0, 98, 0, 67, 0, 70, 0],
        [0, 102, 72, 87, 62, 40, 111, 34, 92, 6, 90, 114, 8, 82, 98, 13, 64, 113, 76, 42, 39, 55, 0, 116, 88, 52, 50, 0,
         20, 0, 94, 119, 61, 10, 0, 59, 100, 77, 57, 81, 69, 95, 4, 22, 60, 68, 53, 0, 17, 103, 18, 9, 99, 44, 28, 19,
         75, 0, 12, 21, 79, 118, 33, 30, 25, 73, 31, 67, 0, 117, 108, 66, 46, 97, 7, 106, 74, 0, 58, 23, 14, 36, 48, 86,
         65, 115, 0, 80, 32, 91, 0, 85, 89, 5, 26, 41, 0, 84, 0, 45, 70, 93, 29, 38, 0, 104, 63, 51, 0, 16, 0, 35, 37,
         24, 1, 56, 11, 27, 0, 107, 71, 96, 54, 112, 3, 78, 109, 43, 2, 49, 0, 47, 101, 0, 110, 83, 15, 0, 105, 0]
    ]
    divided_paths = [
        [0, 11, 8, 18, 0, 3, 2, 23, 0, 22, 9, 0, 4, 28, 0, 27, 29, 15, 10, 25, 5, 20, 0, 12, 1, 7, 16, 26, 24, 0, 30, 0,
         13, 21, 31, 19, 17, 6, 14, 0],
        [0, 17, 12, 3, 6, 25, 0, 40, 29, 43, 13, 38, 9, 36, 22, 0, 2, 41, 14, 0, 19, 24, 33, 21, 32, 42, 37, 5, 0, 31,
         8, 15, 28, 27, 0, 4, 34, 39, 26, 10, 11, 7, 0, 23, 0, 16, 20, 18, 35, 1, 30, 0],
        [0, 48, 22, 10, 30, 53, 0, 42, 45, 5, 54, 2, 58, 0, 24, 1, 36, 44, 49, 31, 28, 0, 59, 38, 52, 0, 7, 29, 13, 8,
         35, 55, 19, 0, 50, 39, 17, 37, 57, 27, 26, 15, 0, 25, 0, 33, 41, 34, 23, 47, 14, 0, 18, 0, 16, 20, 3, 46, 40,
         11, 4, 21, 6, 0, 32, 51, 0, 43, 56, 12, 9, 0],
        [0, 6, 30, 51, 3, 15, 44, 13, 35, 45, 58, 0, 68, 38, 40, 39, 8, 59, 20, 41, 0, 66, 29, 12, 22, 0, 23, 0, 24, 0,
         28, 14, 25, 42, 57, 34, 27, 0, 21, 55, 60, 47, 4, 65, 7, 0, 19, 0, 67, 61, 64, 2, 33, 49, 9, 53, 0, 31, 18, 0,
         63, 11, 46, 5, 52, 26, 54, 0, 37, 50, 16, 10, 17, 32, 43, 0, 56, 62, 48, 1, 36, 0],
        [0, 4, 13, 74, 76, 0, 70, 1, 54, 0, 55, 69, 16, 0, 90, 84, 68, 66, 0, 15, 22, 41, 0, 86, 9, 92, 0, 63, 29, 36,
         88, 57, 64, 34, 0, 37, 6, 65, 14, 40, 0, 80, 43, 48, 26, 0, 59, 67, 87, 0, 28, 25, 10, 7, 0, 3, 77, 44, 72, 0,
         82, 42, 78, 0, 39, 60, 94, 56, 0, 58, 8, 0, 45, 0, 17, 18, 0, 96, 49, 2, 0, 12, 0, 23, 19, 100, 21, 0, 98, 89,
         99, 62, 71, 0, 52, 91, 0, 83, 51, 81, 0, 27, 38, 47, 0, 61, 97, 0, 5, 0, 32, 33, 53, 73, 95, 24, 0, 50, 79, 0,
         20, 35, 46, 0, 31, 0, 93, 75, 0, 11, 85, 30, 0],
        [0, 37, 57, 77, 100, 4, 59, 22, 95, 60, 68, 81, 69, 53, 0, 105, 85, 0, 117, 26, 5, 41, 0, 104, 63, 51, 89, 0,
         47, 101, 0, 96, 71, 116, 54, 118, 79, 21, 30, 67, 33, 38, 52, 88, 50, 107, 61, 119, 94, 10, 0, 80, 16, 70, 55,
         93, 76, 43, 113, 64, 109, 78, 39, 8, 73, 25, 29, 12, 91, 32, 31, 0, 2, 49, 42, 82, 3, 112, 45, 84, 0, 20, 0,
         98, 13, 28, 19, 46, 66, 108, 103, 17, 18, 44, 9, 99, 0, 23, 58, 15, 83, 97, 7, 106, 110, 74, 65, 86, 48, 115,
         36, 0, 75, 14, 0, 102, 72, 87, 62, 40, 111, 34, 27, 11, 56, 1, 24, 35, 92, 6, 114, 90, 0]
    ]

    # define algorithms:
    greedy = Greedy()
    greedy.set_problem(s_maxs[problem_id], max_cars[problem_id], problems[problem_id])
    greedy.best_path = greedy_paths[problem_id]
    greedy.best_number_of_cycles = greedy_number_of_cycles[problem_id]
    greedy.best_cost = greedy_costs[problem_id]

    antColony_basic = AntColony()
    antColony_basic.set_problem(s_maxs[problem_id], max_cars[problem_id], problems[problem_id])
    antColony_basic.best_path = basic_paths[problem_id]
    antColony_basic.best_number_of_cycles = basic_number_of_cycles[problem_id]
    antColony_basic.best_cost = basic_costs[problem_id]
    antColony_basic.iters_done = basic_iters_dones[problem_id]

    antColony_reduced = AntColony_Reduced()
    antColony_reduced.set_problem(s_maxs[problem_id], max_cars[problem_id], problems[problem_id])
    antColony_reduced.best_path = reduced_paths[problem_id]
    antColony_reduced.best_number_of_cycles = reduced_number_of_cycles[problem_id]
    antColony_reduced.best_cost = reduced_costs[problem_id]
    antColony_reduced.iters_done = reduced_iters_dones[problem_id]

    antColony_divided = AntColony_Divided()
    antColony_divided.set_problem(s_maxs[problem_id], max_cars[problem_id], problems[problem_id])
    antColony_divided.best_path = divided_paths[problem_id]
    antColony_divided.best_number_of_cycles = divided_number_of_cycles[problem_id]
    antColony_divided.best_cost = divided_costs[problem_id]
    antColony_divided.iters_done = divided_iters_dones[problem_id]

    save_file = photo_files[problem_id] if is_save_file else None

    # plot solutions:
    plot_4_solutions(greedy, antColony_basic, antColony_reduced, antColony_divided,
                     labels=labels, save_file=save_file, start_title=problem_names[problem_id] + "; ")


# distributions plots:
# those costs are from logs:
greedy_s1_cost = np.array([1547])
greedy_s2_cost = np.array([1412])
greedy_m1_cost = np.array([1902])
greedy_m2_cost = np.array([1575])
greedy_b1_cost = np.array([39789])
greedy_b2_cost = np.array([24634])

basic_s1_costs = np.array([1208, 1209, 1214, 1220, 1225, 1232, 1233, 1236, 1239, 1241, 1248])
basic_s2_costs = np.array([1241, 1250, 1329, 1341, 1342, 1343, 1346, 1355, 1385, 1391, 1427])
basic_m1_costs = np.array([2012, 2083, 2088, 2100, 2113, 2125, 2135, 2135, 2139, 2168, 2180])
basic_m2_costs = np.array([1949, 1966, 1971, 1988, 1993, 1995, 2004, 2008, 2011, 2024, 2033])
basic_b1_costs = np.array([39608, 40147, 40168, 40294, 40335, 40488, 40582, 40736, 40806, 40821, 40892])
basic_b2_costs = np.array([40467, 40844, 41105, 42270, 42836, 42881, 42986, 43160, 43458, 43489, 43740])

reduced_s1_costs = np.array([1291, 1311, 1315, 1317, 1317, 1318, 1318, 1329, 1340, 1346, 1347])
reduced_s2_costs = np.array([1254, 1264, 1268, 1298, 1320, 1321, 1336, 1343, 1343, 1353, 1358])
reduced_m1_costs = np.array([1904, 1930, 1953, 1971, 1983, 2010, 2014, 2019, 2022, 2028, 2032])
reduced_m2_costs = np.array([1697, 1714, 1732, 1732, 1733, 1752, 1768, 1777, 1784, 1789, 1820])
reduced_b1_costs = np.array([35062, 35187, 36051, 36247, 36277, 36397, 36421, 36534, 36595, 36751, 36879])
reduced_b2_costs = np.array([33583, 34467, 34701, 34793, 34998, 35043, 35131, 35193, 35282, 35989, 36002])

divided_s1_costs = np.array([1268, 1270, 1270, 1271, 1271, 1272, 1273, 1273, 1273, 1273, 1273])
divided_s2_costs = np.array([1143, 1150, 1164, 1165, 1169, 1170, 1175, 1176, 1182, 1206, 1206])
divided_m1_costs = np.array([1678, 1687, 1691, 1691, 1693, 1697, 1702, 1707, 1714, 1717, 1729])
divided_m2_costs = np.array([1369, 1386, 1396, 1402, 1407, 1412, 1416, 1422, 1431, 1433, 1449])
divided_b1_costs = np.array([33849, 34193, 34360, 34387, 34471, 34493, 34733, 34781, 34819, 34861, 34865])
divided_b2_costs = np.array([29047, 29273, 29435, 29511, 29789, 29804, 29876, 30204, 30210, 30247, 31449])

s1_costs = [greedy_s1_cost, basic_s1_costs, reduced_s1_costs, divided_s1_costs]
s2_costs = [greedy_s2_cost, basic_s2_costs, reduced_s2_costs, divided_s2_costs]
m1_costs = [greedy_m1_cost, basic_m1_costs, reduced_m1_costs, divided_m1_costs]
m2_costs = [greedy_m2_cost, basic_m2_costs, reduced_m2_costs, divided_m2_costs]
b1_costs = [greedy_b1_cost, basic_b1_costs, reduced_b1_costs, divided_b1_costs]
b2_costs = [greedy_b2_cost, basic_b2_costs, reduced_b2_costs, divided_b2_costs]

costs = [None, s1_costs, s2_costs, m1_costs, m2_costs, b1_costs, b2_costs]
problem_names = [None, "Small 1", "Small 2", "Medium 1", "Medium 2", "Big 1", "Big 2"]

def get_x_for_plot(np_array, m, u=0.3):
    return m + rng.uniform(-1, 1, size=len(np_array)) * u

def get_cords(costs, u=0.3):
    x_cords = np.concatenate([get_x_for_plot(costs[i], i + 1, u) for i in range(1, 4)])
    x_cords = np.concatenate((np.ones(1), x_cords))
    y_cords = np.concatenate(costs)

    return x_cords, y_cords

def plot_distr(problem_id, save=False):
    x_cords, y_cords = get_cords(costs[problem_id])

    colours = ["brown"] + ["green"] * 11 + ["orange"] * 11 + ["red"] * 11

    with plt.style.context('bmh'):
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))#plt.figure(figsize=[2, 1])
        #ax = fig.add_axes([1, 2, 3, 4])

        ax.scatter(x_cords, y_cords, alpha=0.8, c=colours)

        plt.title(problem_names[problem_id])
        plt.xticks([1, 2, 3, 4], ["Greedy", "Basic", "Reduced", "Divided"])

        if save:
            fig.savefig("graphs/distr_{}.png".format(problem_id))