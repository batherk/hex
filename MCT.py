import numpy as np
import random
from Settings import EXPLORATION_FACTOR, POSSIBILITY_FACTOR, EPSILON
from ActorNet import AbstractActorNeuralNet
import math
   
class MonteCarloTree:
    """
    A class that stores states and actions in a tree structure.
    It also chooses actions, given a state.  
    """

    def __init__(self, net:AbstractActorNeuralNet, epsilon=EPSILON, exploration_factor=EXPLORATION_FACTOR, possibility_factor=POSSIBILITY_FACTOR ):
        """Creates a UCT-tree."""
        
        self.states = {}
        self.state_action_pairs = {}
        self.net = net

        self.possibility_factor = possibility_factor
        self.exploration_factor = exploration_factor
        self.epsilon = epsilon
        
    def __contains__(self, state):
        """Returns if the state is in the tree structure."""
        return state in self.states

    def Q(self,state,action):
        """Returns the Q value of a state-action pair."""
        return self.state_action_pairs[(state,action)]["Q"]
    
    def N(self, state, action=None):
        """Returns the number of times a state or a state-action pair is visited."""
        if not action:
            return self.states[state]["N"]
        else:
            return self.state_action_pairs[(state,action)]["N"]

    def default_action(self, game):
        """Returns the default (random) action for a given game in its current state."""
        pre_screen_probabilities = self.net.get_propabilities(game.get_state())
        after_screen_probabilities = game.get_possible_and_normalized_possibilities(pre_screen_probabilities)
        return game.get_action(self.epsilon, after_screen_probabilities)

    def get_max_action_value(self, state, action):
        """Returns the evaluation of an action, given that you are searching for the max value."""
        return self.Q(state,action) + self.get_exploration_bonus(state, action)

    def get_min_action_value(self, state, action):
        """Returns the evaluation of an action, given that you are searching for a minimal value."""
        return self.Q(state,action) - self.get_exploration_bonus(state, action)

    def get_exploration_bonus(self, state, action):
        """Returns the exploration bonus for a given state and action"""
        if self.N(state,action):
            temp = np.log(self.N(state)/self.N(state,action))
        else: 
            temp = np.log(self.N(state)/0.5)
        return self.exploration_factor * np.sqrt(temp)

    def select_action(self, game):
        """Returns the best evaluated action given a game and its current state."""
        state = game.get_state()
        actions = game.get_possible_actions()

        if game.current_player ==1:
            values = [self.get_max_action_value(state,action) for action in actions]
            index = values.index(max(values))
        else:
            values = [self.get_min_action_value(state,action) for action in actions]
            index = values.index(min(values))

        return actions[index]

    def update(self, state, action, result):
        """Updates the values for state and state action pair in the tree structure, based on the result."""
        self.states[state]["N"] += 1
        self.state_action_pairs[(state,action)]["N"] += 1
        self.state_action_pairs[(state,action)]["Q"] += (result-self.Q(state,action))/self.N(state,action)

    def add_state(self, game):
        """Add a state to the tree structure."""
        state = game.get_state()
        actions = game.get_possible_actions()
        self.states[state] = {"N":0,"A":actions}
        for action in actions:
            game_copy = game.copy()
            game_copy.perform_action(action)
            self.state_action_pairs[(state,action)] = {"N":0,"Q":0,"State":game_copy.get_state()}

    def traverse(self, game):
        """Returns a list of the states from the root node to the leaf node."""
        game_copy = game.copy()
        sequence = []
        if game_copy.get_state() not in self:
            raise ValueError("Root state is not in the tree structure")
        while not game_copy.is_done() and game_copy.get_state() in self:
            sequence.append(game_copy.get_state())
            action = self.select_action(game_copy)
            game_copy.perform_action(action)
        sequence.append(game_copy.get_state())
        return sequence

    def print_qs(self):
        """Print all Q(s,a)-values in the tree structure."""
        for state in self.states:
                print(state)
                for action in self.states[state]["A"]:
                    print(action, self.state_action_pairs[(state,action)])
                print()
    
    def clean(self):
        """Removes the stored data."""
        self.states = {}
        self.state_action_pairs = {}

    def set_exploration(self, exploration):
        """Set the exploration constant."""
        self.exploration_factor = exploration

    def reset_exploration(self):
        """Reset the exploration constant."""
        self.exploration_factor = EXPLORATION_FACTOR

    def exploit(self):
        """Setting the exploration constant to 0. Greedingly choosing the next action"""
        self.set_exploration(0)

    def get_possible_actions(self,state):
        return self.states[state]['A']

    def get_normalized_possibilities(self, game):

        state = game.get_state()
        possible_actions = self.get_possible_actions(state)
        possibilities = []
        total = 0

        for action in game.get_all_actions():
            if action in possible_actions:
                if game.current_player == 1:
                    value = math.exp(self.possibility_factor*self.Q(state,action))
                else:
                    value = math.exp(self.possibility_factor*self.Q(state,action)*-1)
                possibilities.append(value)
                total += value
            else:
                possibilities.append(0)
        
        if total == 0:
            print(possibilities)
        
        return [possibility/total for possibility in possibilities]
                



