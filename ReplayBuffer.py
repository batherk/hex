import numpy as np
import json
from Settings import BUFFER_FILENAME, BUFFER_SIZE, ADD_ORIENTATION

class ReplayBuffer:
    
    def __init__(self, max_size=BUFFER_SIZE, filename=BUFFER_FILENAME, add_orientation=ADD_ORIENTATION, clean=False):
        self.filename = "Buffers/" + filename
        self.max_size = max_size
        self.inputs = []
        self.targets = []
        self.saved_index = 0
        self.add_orientation = add_orientation
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

    def get_unsaved_inputs(self):
        inputs = []
        for inp in self.inputs[self.saved_index:]:
            inputs.append(np.array(inp))
            if self.add_orientation:
                inp_orientation = [inp[0]] + list(inp)[len(inp):0:-1]
                inputs.append(np.array(inp_orientation))
        return np.array(inputs)

    def get_unsaved_targets(self):
        targets = []
        for target in self.targets[self.saved_index:]:
            targets.append(np.array(target))
            if self.add_orientation:
                targets.append(np.array(list(target)[::-1]))
        return np.array(targets)

    def get_all_inputs(self):
        inputs = []
        for inp in self.inputs:
            inputs.append(np.array(inp))
            if self.add_orientation:
                inp_orientation = [inp[0]] + list(inp)[len(inp):0:-1]
                inputs.append(np.array(inp_orientation))
        return np.array(inputs)

    def get_all_targets(self):
        targets = []
        for target in self.targets:
            targets.append(np.array(target))
            if self.add_orientation:
                targets.append(np.array(list(target)[::-1]))
        return np.array(targets)

    def add_data(self, input, target):
        self.inputs.append(input)
        self.targets.append(target)
        if len(self.inputs)>self.max_size and self.saved_index!=0:
            self.saved_index -= 1
        self.inputs = self.inputs[-self.max_size:]
        self.targets = self.targets[-self.max_size:]

    def is_empty(self):
        return not bool(len(self.inputs))
    

