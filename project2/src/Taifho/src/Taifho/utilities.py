from string import ascii_lowercase as alphabet
import copy
import re
import numpy as np

# Kroki (move) i pozycje (position) są zapisywane na 2 sposoby. Sposób str, to nais długości 4 dla move i 2 dla position
# sposób int to tablica intów dłógości 4 dla move i 2 dla position.
# najbliższe 4 funkcje to tłumaczenia z str na int i odwrotnie
# Pierwszy int reprezentuje wiersz, drugi kolumnę, czyli drugi tłumaczy się na literę.
# ALE litery pisze się jako pierwsze w wersji str


def move_str_to_int(move_str):
    """
    move_str_to_int("D9D8") == [9, 3, 8, 3]
    move_str_to_int("A9B8") == [9, 0, 8, 1]
    """
    assert (re.match(re.compile("^([a-hA-H][0-9])+"), move_str))

    move_str = move_str.lower()
    move_start_letter = ord(move_str[0]) - 97  # not 96, cos we want a -> 0
    move_start_number = int(move_str[1])
    move_end_letter = ord(move_str[2]) - 97
    move_end_number = int(move_str[3])
    return [move_start_number, move_start_letter, move_end_number, move_end_letter]


def move_int_to_str(move_int):
    """
    move_int_to_str([9, 3, 8, 3]) == "D9D8"
    """
    assert (move_int[0] >= 0)
    assert (move_int[0] <= 9)
    assert (move_int[1] >= 0)
    assert (move_int[1] <= 7)
    assert (move_int[2] >= 0)
    assert (move_int[2] <= 9)
    assert (move_int[3] >= 0)
    assert (move_int[3] <= 7)

    move_str = "" + alphabet[move_int[1]] + str(move_int[0]) + alphabet[move_int[3]] + str(move_int[2])
    return move_str.upper()


def position_str_to_int(position_str):
    """
    position_str_to_int("D9") == [9, 3]
    """
    assert (re.match(re.compile("^([a-hA-H][0-9])"), position_str))

    position_str = position_str.lower()
    position_letter = ord(position_str[0]) - 97  # not 96, cos we want a -> 0
    position_number = int(position_str[1])
    return [position_number, position_letter]


def position_int_to_str(position_int):
    """
    position_int_to_str([9, 3]) == "D9"
    """
    assert (position_int[0] >= 0)
    assert (position_int[0] <= 9)
    assert (position_int[1] >= 0)
    assert (position_int[1] <= 7)

    position_str = "" + alphabet[position_int[1]] + str(position_int[0])
    return position_str.upper()


def next_place(place, direction, steps=1):
    """
    Zwraca następne miejsce w danym kierunku dla bierki z podanej pozycji

    Przykłady:
    next_place([4, 4], 0) == [3 ,4]
    next_place([4, 4], 3) == [5 ,5]
    next_place([0, 0], 1) == None  # the board has ended
    next_place([9, 7], 6) == [9, 6]
    next_place([9, 8], 6)  # Error  # the given place is out of the board

    next_place([4, 4], 0, 3) == [1 ,4]
    next_place([4, 4], 3, 2) == [6, 6]
    next_place([4, 4], 3, 4) == None
    """
    if place[0] < 0 or place[0] > 9 or place[1] < 0 or place[1] > 7:
        raise Exception("the given place is out of the board")
    new_place = copy.copy(place)
    if direction in [0, 1, 7]:
        new_place[0] -= steps
    elif direction in [3, 4, 5]:
        new_place[0] += steps
    if direction in [5, 6, 7]:
        new_place[1] -= steps
    elif direction in [1, 2, 3]:
        new_place[1] += steps
    if new_place[0] < 0 or new_place[0] > 9 or new_place[1] < 0 or new_place[1] > 7:
        return None
    return new_place


def which_move_was_made(board_from, board_to):
    """
    Otrzymuje 2 plansze z rozgrywki, które przesunęły się o połowę tury od siebie (tzn. jeden gracz się poruszył)
    Zwraca ruch, który został wykonany w formie int
    """
    if np.sum(board_from != board_to) != 2:
        raise Exception("Wrong boards passed into `which_move_was_made()` function.")
    differences = np.where(board_from != board_to)  # initial guess of from-to direction
    move_from = [differences[0][0], differences[1][0]]
    move_to = [differences[0][1], differences[1][1]]

    if board_from[move_from[0], move_from[1]] in [0, 9]:  # this place was empty, there couldn't be a move from this place
        move_from, move_to = move_to, move_from

    return [move_from[0], move_from[1], move_to[0], move_to[1]]
