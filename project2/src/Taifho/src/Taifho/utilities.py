from string import ascii_lowercase as alphabet

def move_str_to_int(move_str):
    move_str = move_str.lower()
    move_start_letter = ord(move_str[0]) - 97  # not 96, cos we want a -> 0
    move_start_number = int(move_str[1])
    move_end_letter = ord(move_str[2]) - 97
    move_end_number = int(move_str[3])

    return [move_start_letter, move_start_number, move_end_letter, move_end_number]

def move_int_to_str(move_int):
    move_str = "" + alphabet[move_int[0]] + str(move_int[1]) + alphabet[move_int[2]] + str(move_int[3])
    return move_str.upper()

def position_str_to_int(position_str):
    position_str = position_str.lower()
    position_letter = ord(position_str[0]) - 97  # not 96, cos we want a -> 0
    position_number = int(position_str[1])

    return [position_letter, position_number]

def position_int_to_str(position_int):
    position_str = "" + alphabet[position_int[0]] + str(position_int[1])
    return position_str.upper()