from abc import ABC, abstractmethod
from AbstractGame import AbstractGame
from ActorNet import AbstractActorNeuralNet, LoadedNet, Dense
from ReplayBuffer import ReplayBuffer
from Settings import EPSILON

class AbstractPlayer(ABC):

    def __init__(self, name):
        self.name = name
        self.id = None

    def __str__(self):
        return self.name

    def is_players_turn(self,game:AbstractGame):
        return game.current_player==self.id

    @abstractmethod
    def perform_action(self, game:AbstractGame):
        pass

class NetBotFromTraining(AbstractPlayer):

    def __init__(self, name="Experienced", net:AbstractActorNeuralNet=None, replay_buffer:ReplayBuffer=None, epsilon=EPSILON, train_on_replay_buffer=True):
        super(NetBotFromTraining,self).__init__(name)
        self.epsilon = epsilon
        if net:
            self.net = net
        else: 
            self.net = Dense()
        if not replay_buffer:
            replay_buffer = ReplayBuffer()
        if train_on_replay_buffer:
            self.net.update(replay_buffer.get_all_inputs(),replay_buffer.get_all_targets())

    def perform_action(self, game:AbstractGame):
        pre_screen_probs = self.net.get_propabilities(game.get_state())
        after_screen_probs = game.get_possible_and_normalized_possibilities(pre_screen_probs)
        action = game.get_action(self.epsilon, after_screen_probs)
        game.perform_action(action)

class NetBotFromLoading(AbstractPlayer):

    def __init__(self, filename, name=None, epsilon=EPSILON):
        if not name:
            name = filename
        super(NetBotFromLoading,self).__init__(name)
        self.epsilon = epsilon
        self.net = LoadedNet(filename)

    def perform_action(self, game:AbstractGame):
        pre_screen_probs = self.net.get_propabilities(game.get_state())
        after_screen_probs = game.get_possible_and_normalized_possibilities(pre_screen_probs)
        action = game.get_action(self.epsilon, after_screen_probs)
        game.perform_action(action)

class Human(AbstractPlayer):

    def __init__(self, name):
        super(Human,self).__init__(name)
        
    def perform_action(self, game:AbstractGame):
        while True:
            game.board.show_graph(debug=True)
            user_input = input("Write the spot you want to put your piece (Format y,x): ")
            user_list = user_input.split(",")
            if len(user_list) != 2:
                print("Seperate the numbers with a comma")
                continue
            if not user_list[0].isdigit or not user_list[1].isdigit:
                print("y and x must be numbers")
                continue
            action = int(user_list[0]),int(user_list[1])
            if not action in game.get_possible_actions():
                print("The action is not possible")
                continue
            game.perform_action(action)
            game.board.show_graph()
            break


