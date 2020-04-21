from HexGame import HexGame
from MCT import MonteCarloTree
from copy import deepcopy
from ProgressBar import ProgressBar
from ActorNet import *
from Settings import *
import random
import numpy as np
from ReplayBuffer import ReplayBuffer


class NetTrainer:
    """
    A simulator class that uses UCT to learn and play games.
    """

    def __init__(self, rollout_iterations=ROLLOUT_ITERATIONS, training_iterations=TRAINING_ITERATIONS,playing_iterations=PLAYING_ITERATIONS, starting_player=STARTING_PLAYER_ACTUAL,verbose=VERBOSE, board_size=BOARD_SIZE, net=None, tree=None, replay_buffer=None, train_net_on_init=TRAIN_NET_ON_INIT,load_net=LOAD_NET, cached_nets=CACHED_NETS, train_with_random_samples=TRAIN_WITH_RANDOM_SAMPLES):
        """
        Creates a simulator object. 
        If USE_UI is True then the user can fill in the needed values.
        Else, the settings above the class definition is used.
        """
        self.rollout_iterations = rollout_iterations
        self.playing_iterations = playing_iterations
        self.training_iterations = training_iterations

        self.cached_nets = cached_nets
        self.starting_player_actual = starting_player
        self.train_with_random_samples = train_with_random_samples

        self.verbose = verbose

        self.size = board_size

        if replay_buffer:
            self.replay_buffer = replay_buffer
        else:
            self.replay_buffer = ReplayBuffer(max_size=BUFFER_SIZE, filename=BUFFER_FILENAME)

        if net:
            self.net = net    
        elif load_net: 
            self.net = LoadedNet(DEFAULT_NET)
        else:
            self.net = Dense()
        if train_net_on_init and not self.replay_buffer.is_empty():
            self.net.update(self.replay_buffer.get_all_inputs(), self.replay_buffer.get_all_targets(),EPOCHS_INIT)

        
        if tree:
            self.tree = tree
        else:
            self.tree = MonteCarloTree(self.net)
              
    def create_game(self):
        """Creates a game based on the simulator's attributes."""
        return HexGame(size=self.size, starting_player=self.starting_player_actual)

    def sim_default(self, game):
        """Simulates a game by using the default policy. Returns the first action that is used for learning purposes."""
        first_action = None
        while not game.is_done():
            action = self.tree.default_action(game)
            if not first_action:
                first_action = action
            game.perform_action(action)                
        return first_action

    def sim_tree(self,game):
        """
        Simulates a game using the tree policy.
        It stops when the state is not recognized in the UCT-tree structure.
        Returns a list of state-action pairs that have been visited.
        """
        sequence = []
        while not game.is_done():
            state = game.get_state()
            if state not in self.tree:
                self.tree.add_state(game)
                return sequence
            action = self.tree.select_action(game)
            game.perform_action(action)
            sequence.append((state,action))
        return sequence

    def backprop(self, state_action_sequence, result):
        """Iterates through a list of state-action pairs and updates the UCT-values based on the end result."""
        for state, action in state_action_sequence:
            self.tree.update(state,action,result)

    def simulate_game(self,game):
        """Simulates a game and updates the UCT-tree with the results"""
        state_action_sequence = self.sim_tree(game)
        if not game.is_done():
            state = game.get_state()
            action = self.sim_default(game)
            state_action_sequence.append((state,action))
        self.backprop(state_action_sequence, game.get_end_result())
    
    def simulate_games(self, game):
        """Simulates games given the amount of rollout iterations."""
        times = self.rollout_iterations
        progress = ProgressBar(times, "Choosing move:")
        for i in range(times):
            sim_game = game.copy()
            self.simulate_game(sim_game)
            progress.show(i)
        possibilities = self.tree.get_normalized_possibilities(game)
        return possibilities

    def train_game(self, game):
        """
        Plays a game using preferably the tree policy, but if that's not possible, using the default policy.
        Prints the game's states if verbose is set to True.
        """
        if self.verbose:
            game.board.show_graph(pause=0.00001)

        while not game.is_done():
            self.tree.clean()
            possibilities = self.simulate_games(game)

            self.replay_buffer.add_data(game.get_state(),possibilities)
            action = game.get_action(self.tree.epsilon, possibilities)
            game.perform_action(action)

            if self.verbose:
                game.board.show_graph(pause=0.00001)
        
            self.tree.clean()
        all_inputs = self.replay_buffer.get_all_inputs()
        all_targets = self.replay_buffer.get_all_targets()

        if self.train_with_random_samples:
            indexes_random_samples = [random.randint(0,len(all_inputs)-1) for i in range(SAMPLES_WHILE_TRAINING)]
            inputs = []
            targets = []
            
            for index in indexes_random_samples:
                inputs.append(all_inputs[index])
                targets.append(all_targets[index])
            self.net.update(np.array(inputs), np.array(targets), EPOCHS_WHILE_TRAINING)
        else: 
            self.net.update(np.array(all_inputs), np.array(all_targets), EPOCHS_WHILE_TRAINING)

        self.replay_buffer.save_data()
               
    def train_games(self):
        """
        Plays the amount of games that is given by training_iterations.
        If self.verbose is set to False, the progress of the games played will be shown.
        Lastly, the result statistics of the games played will be printed. 
        """
        for i in range(1, self.training_iterations+1):
            print(f"Training game {i} of {self.training_iterations}")
            game = self.create_game()
            self.train_game(game)
            if self.cached_nets > 0 and i%(self.training_iterations//self.cached_nets) == 0:
                self.net.save(f"After_{i}")

    def play_game(self, game):
        """
        Plays a game using preferably the tree policy, but if that's not possible, using the default policy.
        Prints the game's states if verbose is set to True.
        """
        if self.verbose:
            game.board.show_graph()

        while not game.is_done():
            action = self.tree.default_action(game)
            game.perform_action(action)

            if self.verbose:
                game.board.show_graph()

    def play_games(self):
        for i in range(self.playing_iterations):
            game = self.create_game()
            self.play_game(game)
            
    def test_game(self,game):
        if self.verbose:
            game.board.show_graph(pause=0.001)
        
        iteration = 1

        while not game.is_done():
            print(f"Move number {iteration}")

            possibilities = self.simulate_games(game)
            pre_screen_probabilities = self.net.get_propabilities(game.get_state())
            after_screen_probabilities = game.get_possible_and_normalized_possibilities(pre_screen_probabilities)

            action_net = game.get_action(self.tree.epsilon, after_screen_probabilities)
            action_tree = game.get_action(self.tree.epsilon, possibilities)
            actions = game.get_all_actions()

            for i in range(len(possibilities)):
                print(f"Move: {actions[i]}. MCTS: {possibilities[i]:.2f}. NET: {pre_screen_probabilities[i]:.2f}. After: {after_screen_probabilities[i]:.2f}")

            print(f"Tree action: {action_tree}. Net action: {action_net}.")
            game.perform_action(action_tree)

            iteration += 1

            if self.verbose:
                game.board.show_graph(pause=0.001)

    def test_games(self):
        for i in range(self.playing_iterations):
            game = self.create_game()
            self.test_game(game)