import random
from tensorflow.keras.activations import relu, tanh, sigmoid, linear, softmax
from tensorflow.keras.optimizers import SGD, RMSprop, Adagrad, Adam

from HexGame import HexGame
from ProgressBar import ProgressBar
from NetTrainer import NetTrainer
from ReplayBuffer import ReplayBuffer
from ActorNet import Dense, LoadedNet
from Player import Human, NetBotFromLoading, NetBotFromTraining, RandomPlayer
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
    gt = NetTrainer(load_net=True)
    gt.play_games()
elif RUNS[RUN] == "Testing MCTS- vs Net-probabilities": 
    print("Mode: Testing probabilities when the model play against itself.")
    gt = NetTrainer()
    gt.test_games()
elif RUNS[RUN] == "Match with loaded nets": 
    print("Mode: Match. Two players play against each other after loading saved nets.")
    game = HexGame()
    if len(NETS_MATCH)!=2:
        raise ValueError("There must be two nets to play a match")
    player1 = NetBotFromLoading(NETS_MATCH[0])
    player2 = NetBotFromLoading(NETS_MATCH[1])
    match = Match(game, player1, player2)
    match.play_games()
elif RUNS[RUN] == "Match vs random":
    print("Mode: Match. Loaded default against random player")
    game = HexGame()
    player1 = NetBotFromLoading(DEFAULT_NET)
    player2 = RandomPlayer()

    match = Match(game, player1, player2)
    match.play_games()
elif RUNS[RUN] == "Tournament - different net structures using replay buffer": 
    print("Mode: Tournament. Several players play against each other.")
    game = HexGame()
    replay_buffer = ReplayBuffer()
    net1 = Dense(hidden_layers=[(100,relu)], optimizer=Adam)
    net2 = Dense(hidden_layers=[(100,sigmoid)], optimizer=Adam)
    net3 = Dense(hidden_layers=[(100,relu)], optimizer=SGD)
    net4 = Dense(hidden_layers=[(100,sigmoid)], optimizer=SGD)
    player1 = NetBotFromTraining("Adam relu", net1, replay_buffer)
    player2 = NetBotFromTraining("Adam sig", net2, replay_buffer)
    player3 = NetBotFromTraining("SGD relu", net3, replay_buffer)
    player4 = NetBotFromTraining("SGD sig", net4, replay_buffer)

    players = [player1, player2, player3, player4]

    tournament = Tournament(game,players)
    wins = tournament.play_tournament()
elif RUNS[RUN] == "Tournament - different training amounts from loading ": 
    print("Mode: Trained tournament. Already trained nets play against each other.")
    game = HexGame()
    players = [NetBotFromLoading(name) for name in NETS_TOURNAMENT]
    tournament = Tournament(game,players)
    wins = tournament.play_tournament()
elif RUNS[RUN] == "Train net on buffer and save it": 
    print("Mode: Train net on buffer and save it")
    trainer = NetTrainer(train_net_on_init=True)
    trainer.net.save("Saved_net")
else: 
    print("Custom - nothing")
    
