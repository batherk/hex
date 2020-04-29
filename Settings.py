from tensorflow.keras.activations import relu, tanh, sigmoid, linear, softmax
from tensorflow.keras.optimizers import SGD, RMSprop, Adagrad, Adam


# Run
RUNS = {0:"Custom",
        1:"Training model", 
        2:"Playing model against itself", 
        3:"Testing MCTS- vs Net-probabilities", 
        4:"Match with loaded nets", 
        5:"Match after training nets",
        6:"Match vs random",
        7:"Tournament - different net structures using replay buffer", 
        8:"Tournament - different training amounts while training", 
        9:"Tournament - different training amounts from loading ",
        10:"Train net on buffer and save it"}

RUN = 9

# Game
BOARD_SIZE = 6

# Simulation
ROLLOUT_ITERATIONS = 1000
STARTING_PLAYER_ACTUAL = 3

# Training
TRAINING_ITERATIONS = 300
TRAIN_WITH_RANDOM_SAMPLES = False
SAMPLES_WHILE_TRAINING = 50
EPOCHS_WHILE_TRAINING = 1000
CACHED_NETS = 10

# Playing
PLAYING_ITERATIONS = 1000

# Replay buffer
BUFFER_FILENAME = f"{BOARD_SIZE}x{BOARD_SIZE}/training.json"
BUFFER_SIZE = 200
ADD_ORIENTATION = True

# Tree
EPSILON = 0.2
EXPLORATION_FACTOR = 1
POSSIBILITY_FACTOR = 10

# Net
DEFAULT_NET = "After_100"
HIDDEN_LAYERS = [(1000,relu),(1000,relu),(1000,relu)]
OPTIMIZER = SGD
LEARNING_RATE = 0.001
EPOCHS_INIT = 1000
LOAD_NET = False
TRAIN_NET_ON_INIT = True

# Visualization
VERBOSE = False
PAUSE = 0.1

# Tournament
NET_BOTS = ["After_30", "After_60"]