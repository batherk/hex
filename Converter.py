from ReplayBuffer import ReplayBuffer

def binary_to_int(state):
    new_state = [state[0]]
    for i in range(1,len(state),2):
        if (state[i],state[i+1]) == (0,0):
            new_state.append(0)
        elif (state[i],state[i+1]) == (0,1):
            new_state.append(1)
        elif (state[i],state[i+1]) == (1,0):
            new_state.append(2)
        else:
            raise ValueError("Not possible state")
    return new_state

def create_int_from_binary(new_filename):
    rb = ReplayBuffer()
    rb_int = ReplayBuffer(filename=new_filename)

    for i, state in enumerate(rb.inputs):
        rb_int.add_data(binary_to_int(state),rb.targets[i])
    rb_int.save_data()
