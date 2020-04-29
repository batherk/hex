import numpy as np
import json
from Settings import BUFFER_FILENAME, BUFFER_SIZE, ADD_ORIENTATION

class ReplayBuffer:
    
    def __init__(self, max_size=BUFFER_SIZE, filename=BUFFER_FILENAME, clean=False):
        self.filename = "Buffers/" + filename
        self.max_size = max_size
        self.inputs = []
        self.targets = []
        self.saved_index = 0
        if clean:
            self.clear()
        else:
            self.load_data()

    def load_data(self):
        try: 
            file = open(self.filename, 'r')
            datastore = json.load(file)
            self.inputs = datastore['inputs']
            self.targets = datastore['targets']
            self.saved_index = len(self.inputs)
        except FileNotFoundError:
            self.save_data()

    def save_data(self):
        file = open(self.filename, 'w')
        json.dump({"inputs":self.inputs, "targets":self.targets}, file)
        self.saved_index = len(self.inputs)
        file.close()
    
    def clear(self):
        self.inputs = []
        self.targets = []
        self.save_data()

    def get_all_inputs(self):
        return self.inputs

    def get_all_targets(self):
        return self.targets

    def add_data(self, input, target):
        self.inputs.append(input)
        self.targets.append(target)
        if len(self.inputs)>self.max_size and self.saved_index!=0:
            self.saved_index -= 1
        self.inputs = self.inputs[-self.max_size:]
        self.targets = self.targets[-self.max_size:]

    def is_empty(self):
        return not bool(len(self.inputs))
    

