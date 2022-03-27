from .ReadCSV import readCSV


class Problem:
    def __init__(self, file=None, coordinate=None, request=None, capacity=None):
        if file is not None:
            coordinate, request, capacity = readCSV(file)

        self.coordinate = coordinate
        self.request = request
        self.capacity = capacity

    def get_data(self):
        return self.coordinate, self.request, self.capacity
