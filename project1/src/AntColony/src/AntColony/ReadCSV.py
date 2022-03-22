import numpy as np
import pandas as pd

from bs4 import BeautifulSoup

from os import listdir
from os.path import isfile, join, isdir


def readCSV(file):
    if not isfile(file):
        return None

    with open(file, 'r') as f:
        data = f.read()

    Bs_data = BeautifulSoup(data, "xml")

    b_nodes = Bs_data.find_all('node')
    b_requests = Bs_data.find_all('request')

    starting_coordinates = np.array([float(b_nodes[0].find("cx").text),
                                     float(b_nodes[0].find("cy").text)])

    coordinates = np.empty((len(b_nodes) - 1, 2))
    coordinates[:, 0] = np.array([float(b_nodes[i].find("cx").text) for i in range(1, len(b_nodes))])
    coordinates[:, 1] = np.array([float(b_nodes[i].find("cy").text) for i in range(1, len(b_nodes))])

    request = np.array([float(b_requests[i].find("quantity").text) for i in range(len(b_requests))])

    capacity = float(Bs_data.find('capacity').text)

    coordinates = np.concatenate((np.array(starting_coordinates).reshape(1, 2), np.array(coordinates)))

    return coordinates, request, capacity