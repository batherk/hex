import tensorflow as tf
import numpy as np
from tensorflow.keras.activations import softmax
from Settings import BOARD_SIZE, HIDDEN_LAYERS, OPTIMIZER, LEARNING_RATE
from abc import ABC, abstractmethod
from Settings import EPOCHS_INIT


class AbstractActorNeuralNet(ABC):

    def __init__(self, board_size=BOARD_SIZE):
        self.board_size = board_size
        self.model = None        

    def update(self, states, target_possibilities, epochs=EPOCHS_INIT):
        return self.model.fit(states,target_possibilities, epochs=epochs)

    def get_propabilities(self, state):
        return tuple(self.model(np.array([tf.dtypes.cast(np.array(state),dtype=tf.int32)]))[0])

    def save(self,filename):
        self.model.save(f"Models/{self.board_size}x{self.board_size}/" + filename)


class LoadedNet(AbstractActorNeuralNet):
    def __init__(self,filename,board_size=BOARD_SIZE):
        super(LoadedNet,self).__init__(board_size)
        self.model = tf.keras.models.load_model(f"Models/{self.board_size}x{self.board_size}/" + filename)


class Dense(AbstractActorNeuralNet):
    def __init__(self,hidden_layers=HIDDEN_LAYERS,board_size=BOARD_SIZE,optimizer=OPTIMIZER, learning_rate=LEARNING_RATE):
        super(Dense,self).__init__(board_size)

        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Dense(board_size**2+1))
        for node_count, activation in hidden_layers:
            self.model.add(tf.keras.layers.Dense(node_count,activation=activation))
        self.model.add(tf.keras.layers.Dense(board_size**2, activation=softmax))
        self.model.compile(optimizer=optimizer(lr=learning_rate),loss='categorical_crossentropy', metrics=['accuracy'])

        