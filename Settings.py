from tensorflow.keras.activations import relu, tanh, sigmoid, linear, softmax
from tensorflow.keras.optimizers import SGD, RMSprop, Adagrad, Adam


# Run
RUNS = {0:"Custom",
        1:"Training model", 
        2:"Playing model against itself", 
        3:"Testing MCTS- vs Net-probabilities", 
        4:"Match with loaded nets", 
        5:"Match vs random",
        6:"Tournament - different net structures using replay buffer", 
        7:"Tournament - different training amounts from loading ",
        8:"Train net on buffer and save it"}

RUN = 8

# Game
BOARD_SIZE = 6

# Simulation
ROLLOUT_ITERATIONS = 25
STARTING_PLAYER_ACTUAL = 3

# Training
TRAINING_ITERATIONS = 100
TRAIN_WITH_RANDOM_SAMPLES = False
SAMPLES_WHILE_TRAINING = 100
EPOCHS_WHILE_TRAINING = 100
EPOCHS_INIT = 300
LOAD_NET = False
TRAIN_NET_ON_INIT = False
CACHED_NETS = 5

# Playing
PLAYING_ITERATIONS = 10

# Replay buffer
BUFFER_FILENAME = f"{BOARD_SIZE}x{BOARD_SIZE}/experienced.json"
BUFFER_SIZE = 100
ADD_ORIENTATION = True

# Tree
EPSILON = 1
EXPLORATION_FACTOR = 1
POSSIBILITY_FACTOR = 10

# Net
DEFAULT_NET = "After_3_340"
HIDDEN_LAYERS = [(1000,relu),(1000,relu),(1000,relu)]
OPTIMIZER = SGD
LEARNING_RATE = 0.001

# Visualization
VERBOSE = False
PAUSE = 0.1

# Match
NETS_MATCH = ["After_3_340","New"]

# Tournament
NETS_TOURNAMENT = ["After_3_340","New"]
ADD_RANDOM_TO_TOURNAMENT = True