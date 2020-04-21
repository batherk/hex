from tensorflow.keras.activations import relu, tanh, sigmoid, linear, softmax
from tensorflow.keras.optimizers import SGD, RMSprop, Adagrad, Adam


# Run
RUNS = {0:"Custom",
        1:"Training model", 
        2:"Playing model against itself", 
        3:"Testing MCTS- vs Net-probabilities", 
        4:"Match with loaded nets", 
        5:"Match after training nets", 
        6:"Tournament - different net structures using replay buffer", 
        7:"Tournament - different training amounts while training", 
        8:"Tournament - different training amounts from loading "}

RUN = 4

# Game
BOARD_SIZE = 4

# Simulation
ROLLOUT_ITERATIONS = 100
STARTING_PLAYER_ACTUAL = 3

# Training
TRAINING_ITERATIONS = 4
SAMPLES_WHILE_TRAINING = 50
CACHED_NETS = 4

# Playing
PLAYING_ITERATIONS = 1000

# Replay buffer
BUFFER_FILENAME = f"{BOARD_SIZE}x{BOARD_SIZE}/experienced.json"
BUFFER_SIZE = 1000

# Tree
EPSILON = 1
EXPLORATION_FACTOR = 2
POSSIBILITY_FACTOR = 10

# Net
DEFAULT_NET = "Experienced"
HIDDEN_LAYERS = [(100,sigmoid)]
OPTIMIZER = Adam
LEARNING_RATE = 0.01
EPOCHS_INIT = 300
EPOCHS_WHILE_TRAINING = 1
TRAIN_NET_ON_INIT = False

# Visualization
VERBOSE = False
PAUSE = 0.1