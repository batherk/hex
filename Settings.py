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

RUN = 1

# Game
BOARD_SIZE = 6

# Simulation
ROLLOUT_ITERATIONS = 100
STARTING_PLAYER_ACTUAL = 3

# Training
TRAINING_ITERATIONS = 100
TRAIN_WITH_RANDOM_SAMPLES = False
SAMPLES_WHILE_TRAINING = 100
EPOCHS_WHILE_TRAINING = 200
CACHED_NETS = 5

# Playing
PLAYING_ITERATIONS = 1000

# Replay buffer
BUFFER_FILENAME = f"{BOARD_SIZE}x{BOARD_SIZE}/experienced.json"
BUFFER_SIZE = 3000

# Tree
EPSILON = 1
EXPLORATION_FACTOR = 1
POSSIBILITY_FACTOR = 10

# Net
DEFAULT_NET = "After_3_340"
HIDDEN_LAYERS = [(100,sigmoid)]
OPTIMIZER = Adam
LEARNING_RATE = 0.01
EPOCHS_INIT = 300
LOAD_NET = True
TRAIN_NET_ON_INIT = False

# Visualization
VERBOSE = False
PAUSE = 0.1