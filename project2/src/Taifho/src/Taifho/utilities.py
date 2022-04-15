from string import ascii_lowercase as alphabet

# first int represents a letter, so the order in int coding is different from the order in str coding
def move_str_to_int(move_str):
    move_str = move_str.lower()
    move_start_letter = ord(move_str[0]) - 97  # not 96, cos we want a -> 0
    move_start_number = int(move_str[1])
    move_end_letter = ord(move_str[2]) - 97
    move_end_number = int(move_str[3])

    return [move_start_number, move_start_letter, move_end_number, move_end_letter]

def move_int_to_str(move_int):
    move_str = "" + alphabet[move_int[1]] + str(move_int[0]) + alphabet[move_int[3]] + str(move_int[2])
    return move_str.upper()

def position_str_to_int(position_str):
    position_str = position_str.lower()
    position_letter = ord(position_str[0]) - 97  # not 96, cos we want a -> 0
    position_number = int(position_str[1])

    return [position_number, position_letter]

def position_int_to_str(position_int):
    position_str = "" + alphabet[position_int[1]] + str(position_int[0])
    return position_str.upper()

def next_place(place, direction, steps=1):
    """
    Returns the next place from the given one

    Exp:
    next_place([4, 4], 0) == [3 ,4]
    next_place([4, 4], 3) == [5 ,5]
    next_place([0, 0], 1) == None  # the board has ended
    next_place([9, 7], 6) == [9, 6]
    next_place([9, 8], 6)  # Error  # the given place is out of the board

    next_place([4, 4], 0, 3) == [1 ,4]
    next_place([4, 4], 3, 2) == [6, 6]
    next_place([4, 4], 3, 4) == None
    """

    # TODO
    pass