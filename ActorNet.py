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
        self.cast = False        

    def update(self, input_tensors, target_tensors, epochs=EPOCHS_INIT):
        return self.model.fit(input_tensors,target_tensors, epochs=epochs)

    def get_propabilities(self, state):
        if not self.cast:
            try: 
                return tuple(self.model(np.array([np.array(state)]))[0])
            except ValueError:
                self.cast = True
        return tuple(self.model(np.array([tf.cast(np.array(state),tf.int64)]))[0])


    def save(self,filename):
        self.model.save(f"Models\\{self.board_size}x{self.board_size}\\" + filename)


class LoadedNet(AbstractActorNeuralNet):
    def __init__(self,filename,board_size=BOARD_SIZE):
        super(LoadedNet,self).__init__(board_size)
        self.model = tf.keras.models.load_model(f"Models\\{self.board_size}x{self.board_size}\\" + filename)


class Dense(AbstractActorNeuralNet):
    def __init__(self,hidden_layers=HIDDEN_LAYERS,board_size=BOARD_SIZE,optimizer=OPTIMIZER, learning_rate=LEARNING_RATE):
        super(Dense,self).__init__(board_size)

        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Dense(board_size**2+1))
        for node_count, activation in hidden_layers:
            self.model.add(tf.keras.layers.Dense(node_count,activation=activation))
        self.model.add(tf.keras.layers.Dense(board_size**2, activation=softmax))
        self.model.compile(optimizer=optimizer(lr=learning_rate),loss='categorical_crossentropy', metrics=['accuracy'])

        