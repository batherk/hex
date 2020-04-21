from HexGrid import Diamond
from AbstractGame import AbstractGame
from copy import deepcopy
from Settings import BOARD_SIZE
import random

class HexGame(AbstractGame):

    def __init__(self,  size=BOARD_SIZE, starting_player=1):
        self.board = Diamond(size=size)
        self.size = size
        self.visited_pos = {1:[], 2:[]}
        self.winner = 0
        super(HexGame,self).__init__(starting_player)

    def is_done(self):
        """Returns if the game is won or not"""
        return bool(self.winner)

    def copy(self):
        """Returns a deep copy of itsself"""
        copy = HexGame(self.size)
        copy.current_player = self.current_player
        copy.board.set_state(self.board.get_state())
        copy.visited_pos = deepcopy(self.visited_pos)
        return copy

    def get_state(self):
        """Gets the current state of the game"""
        return (self.current_player,) + self.board.get_state()

    def perform_action(self, action):
        """¨Updates the current state by performing the action."""
        if action not in self.get_possible_actions():
            print(action, self.get_state())
            raise ValueError(f"Not legal action {action}")
        self.board.fill_node(action, self.current_player)
        self.add_pos_to_visited_list(action, self.current_player)

        self.switch_player()

    def get_possible_actions(self):
        """¨Returns a list of the possible actions to perform given the game's current state"""

        if not self.board.get_empty_nodes_positions() and not self.is_done():
            raise ValueError("Game is done")
        return self.board.get_empty_nodes_positions()
    
    def add_pos_to_visited_list(self, pos, value):
        queue = self.board.get_neighbor_pos_with_same_value(pos)
        if bool(set(self.visited_pos[value])&set(queue)) or (value == 1 and pos[1] == 0) or (value == 2 and pos[0] == 0):
            if value == 1 and pos[1] == self.size-1 or value == 2 and pos[0] == self.size-1:
                self.winner = value
                return
            elif pos not in self.visited_pos[value]:
                self.visited_pos[value].append(pos)

            while len(queue)>0:
                current_pos = queue.pop()
                if value == 1 and current_pos[1] == self.size-1 or value == 2 and current_pos[0] == self.size-1:
                        self.winner = value
                        return
                for neighbor_pos in self.board.get_neighbor_pos_with_same_value(current_pos):
                    if neighbor_pos not in queue and neighbor_pos not in self.visited_pos[value]:
                        queue.append(neighbor_pos)
                if current_pos not in self.visited_pos[value]:
                    self.visited_pos[value].append(current_pos)

    def get_possible_and_normalized_possibilities(self,possibilities):
        if len(possibilities)!= self.size**2:
            raise ValueError(f"Possibilities must have the same length as positions in the board. Length: {len(possibilities)}. Expected length:{self.size**2} ")
        total = 0 
        state = self.board.get_state()
        pos = list(deepcopy(possibilities))

        for i in range(len(pos)):
            if state[2*i:2*i+2] == (0,0):  
                total += pos[i]
            else: 
                pos[i] = 0

        if total == 0:
            print(self)
        
        return tuple([possibility/total for possibility in pos])

    def get_all_actions(self):
        return self.board.get_legal_positions()

    def __str__(self):
        return str(self.get_state())

    def get_action(self, epsilon, possibilities):
        if random.random() < epsilon:
            return random.choices(self.get_all_actions(), weights=possibilities)[0]
        else:
            return self.get_all_actions()[possibilities.index(max(possibilities))]

    def set_state(self, state):
        self.current_player = state[0]
        self.board.set_state(state[1:])
        for pos in self.board.get_filled_positions():
            self.add_pos_to_visited_list(pos, self.board.get_value(pos))
