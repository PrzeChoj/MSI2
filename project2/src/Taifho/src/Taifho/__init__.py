"""
Taifho Python package
"""

from .Position import Position
from .utilities import *

__all__ = ['Position',
           'move_str_to_int', 'move_int_to_str', 'position_str_to_int', 'position_int_to_str', 'next_place', 'which_move_was_made']
