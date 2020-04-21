import random
from tensorflow.keras.activations import relu, tanh, sigmoid, linear, softmax
from tensorflow.keras.optimizers import SGD, RMSprop, Adagrad, Adam

from HexGame import HexGame
from ProgressBar import ProgressBar
from NetTrainer import NetTrainer
from ReplayBuffer import ReplayBuffer
from ActorNet import Dense, LoadedNet
from Player import Human, NetBotFromLoading, NetBotFromTraining
from MCT import MonteCarloTree
from Settings import *
from Match import Match
from Tournament import Tournament

if RUN not in RUNS:
    print("This is not a predefined mode.")
elif RUNS[RUN] == "Training model":
    print("Mode: Training model and generating training data.")
    gt = NetTrainer()
    gt.train_games()
elif RUNS[RUN] == "Playing model against itself": 
    print("Mode: Playing the model against itself.")
    gt = NetTrainer()
    gt.play_games()
elif RUNS[RUN] == "Testing MCTS- vs Net-probabilities": 
    print("Mode: Testing probabilities when the model play against itself.")
    gt = NetTrainer()
    gt.test_games()
elif RUNS[RUN] == "Match with loaded nets": 
    print("Mode: Match. Two players play against each other after loading saved nets.")
    game = HexGame()
    player1 = NetBotFromLoading("Experienced")
    player2 = NetBotFromLoading("Challenger")

    match = Match(game, player1, player2)
    match.play_games()
elif RUNS[RUN] == "Match after training nets":
    print("Mode: Match. Two players play against each other after training nets on the same buffer.")
    game = HexGame()
    pb = ReplayBuffer()
    exp_net = Dense(hidden_layers=[(100,relu)],optimizer=SGD)
    chal_net = Dense()
    player1 = NetBotFromTraining("Experienced",exp_net)
    player2 = NetBotFromTraining("Challenger",chal_net)

    match = Match(game, player1, player2)
    match.play_games()
elif RUNS[RUN] == "Tournament - different net structures using replay buffer": 
    print("Mode: Tournament. Several players play against each other.")
    game = HexGame()
    replay_buffer = ReplayBuffer()
    net1 = Dense(hidden_layers=[(100,relu)], optimizer=Adam)
    net2 = Dense(hidden_layers=[(100,tanh)], optimizer=Adam)
    net3 = Dense(hidden_layers=[(100,sigmoid)], optimizer=Adam)
    net4 = Dense(hidden_layers=[(100,softmax)], optimizer=Adam)
    net5 = Dense(hidden_layers=[(100,linear)], optimizer=Adam)
    player1 = NetBotFromTraining("RELU", net1, replay_buffer)
    player2 = NetBotFromTraining("Tanh", net2, replay_buffer)
    player3 = NetBotFromTraining("Sigmoid", net3, replay_buffer)
    player4 = NetBotFromTraining("SoftMax", net4, replay_buffer)
    player5 = NetBotFromTraining("Linear", net5, replay_buffer)

    players = [player1, player2, player3, player4, player5]

    tournament = Tournament(game,players)
    wins = tournament.play_tournament()
elif RUNS[RUN] == "Tournament - different training amounts while training": 
    print("Mode: Training tournament. A neural net is trained for a number of games. Models from different phases of the training play against each other.")
    game = HexGame()
    net = Dense()
    replay_buffer = ReplayBuffer(3000,f"{BOARD_SIZE}x{BOARD_SIZE}/tt.json", clean=True)
    players = []
    gt = NetTrainer(net=net,replay_buffer=replay_buffer, train_net_on_init=False)
    gt.train_games()
    for i in range(1, gt.cached_nets+1):
        filename_net = f"After_{i*(gt.training_iterations//gt.cached_nets)}"
        new_net = LoadedNet(filename=filename_net)
        players.append(NetBotFromLoading(filename=filename_net, name=f"trained {i*(gt.training_iterations//CACHED_NETS)}"))
    tournament = Tournament(game,players)
    wins = tournament.play_tournament()
elif RUNS[RUN] == "Tournament - different training amounts from loading ": 
    print("Mode: Trained tournament. Already trained nets play against each other.")
    game = HexGame()
    
    player1 = NetBotFromLoading("After_2")
    player2 = NetBotFromLoading("After_8")
    player3 = NetBotFromLoading("After_20")
    player4 = NetBotFromLoading("After_40")
    player5 = NetBotFromLoading("After_50")
    player6 = NetBotFromLoading("Experienced")

    players = [player1, player2, player3, player4, player5, player6]

    tournament = Tournament(game,players)
    wins = tournament.play_tournament()
else: 
    print("Custom - nothing")
    #trainer = NetTrainer(train_net_on_init=True)
    #trainer.net.save("Challenger")
