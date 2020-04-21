import networkx as nx
import matplotlib.pyplot as plt
from Settings import PAUSE

class Node:
    """ Used to store value of individual place on the board, and the relations to other places"""

    def __init__(self, value=0):
        self.value = value
        self.neighbors = {}


    def add_neighbor(self,position,node):
        if position[0]==0 and position[1]==0:
            raise Exception("Cannot add a neighbor with same location")
        for coord in position:
            if coord not in [-1,0,1]:
                raise Exception("Not a legal position")
        if position in self.neighbors:
            raise Exception("Place already taken")

        self.neighbors[position]=node    

    def get_filled_neighbors(self):
        return [node for node in self.get_all_neighbors() if node.value]

    def get_empty_neighbors(self):
        return [node for node in self.get_all_neighbors() if not node.value]

    def get_all_neighbors_with_same_value(self):
        return [node for node in self.get_all_neighbors() if node.value==self.value]

    def get_all_neighbors(self):
        return list(self.neighbors.values())

    def is_neighbor_to(self, other_node):
        return other_node in self.get_all_neighbors()

    def get_relative_pos(self,other_node):
        for key in self.neighbors:
            if self.neighbors[key] == other_node:
                return key
        raise Exception("Nodes are not neighbors")
    

    def __str__(self):
        return str(self.value)

class HexGrid:
    """ Grid of nodes. Used to store the game state and perform actions"""

    def __init__(self, *args, **kwargs):
        self.nodes = {}
        self.positions = {}
        self.graph = nx.Graph()
        self.size = 0

    def get_node(self,pos):
        if pos not in self.get_legal_positions():
            raise ValueError(f"Not legal position: {pos}")
        return self.nodes[pos]

    def fill_node(self, pos, value=1):
        self.nodes[pos].value = value

    def clear_node(self, pos):
        self.nodes[pos].value = 0

    def get_value(self,pos):
        return self.nodes[pos].value

    def get_position(self, node):
        return self.positions[node]

    def get_positions(self,nodes):
        return [self.get_position(node) for node in nodes]
    
    def get_all_nodes(self):
        return list(self.nodes.values())

    def get_positions_with_value(self, value):
        return [node_pos for node_pos in self.nodes if self.nodes[node_pos].value==value]

    def get_empty_nodes_positions(self):
        return self.get_positions_with_value(0)

    def get_legal_positions(self):
        return list(self.nodes.keys())

    def add_node(self,position):
        new_node = Node()
        self.nodes[position] = new_node
        self.positions[new_node]=position
        self.graph.add_node(position)

    def add_edge(self,pos1,pos2):
        relative_pos = (pos2[0]-pos1[0],pos2[1]-pos1[1])
        self.nodes[pos1].add_neighbor(relative_pos,self.nodes[pos2])
        self.graph.add_edge(pos1,pos2)

    def show_graph(self, positions=None, debug=False, pause=PAUSE, action_nodes_pos=[]):
        """
        Show a graph representation of the board/hex grid.
        action_nodes_pos is a list of node positions that should be highlighted (in green)
        """
        if not positions:
            positions = {}
            labels = {}
            label_positions = {}
            for pos in self.get_legal_positions():
                positions[pos]=(pos[1],self.size - pos[0])
                labels[pos]=str(pos)
                label_positions[pos] = (pos[1],self.size - pos[0]-0.3)

        node_pos_with_value_1 = self.get_positions_with_value(1)
        node_pos_with_value_2 = self.get_positions_with_value(2)
        
        for node_pos in action_nodes_pos:
            for player_pos in [node_pos_with_value_1,node_pos_with_value_2]:
                if node_pos in player_pos:
                    player_pos.remove(node_pos)

        nx.draw_networkx_nodes(self.graph, positions, nodelist=self.get_empty_nodes_positions(), node_color='black')
        nx.draw_networkx_nodes(self.graph, positions, nodelist=node_pos_with_value_1, node_color='blue')
        nx.draw_networkx_nodes(self.graph, positions, nodelist=node_pos_with_value_2, node_color='red')
        nx.draw_networkx_nodes(self.graph, positions, nodelist=action_nodes_pos, node_color='green')
        nx.draw_networkx_edges(self.graph, positions, alpha=0.5, width=1)

        if debug:
            nx.draw_networkx_labels(self.graph, label_positions,labels=labels)
        plt.axis('off')
        plt.draw()
        plt.pause(pause)
        plt.clf()

    def is_neighbors(self, node1_pos, node2_pos):
        return self.get_node(node1_pos).is_neighbor_to(self.get_node(node2_pos))

    def get_state(self):
        output = []
        for pos in self.nodes:
            if self.nodes[pos].value == 0:
                output += [0,0]
            elif self.nodes[pos].value == 1:
                output += [0,1]
            elif self.nodes[pos].value == 2:
                output += [1,0]
            else:
                output += [1,1]
        return tuple(output)

        return tuple([])

    def set_state(self, state):
        for i,pos in enumerate(self.nodes):
            if state[2*i:2*i + 2] == (0,0):
                self.nodes[pos].value = 0
            elif state[2*i:2*i + 2] == (0,1):
                self.nodes[pos].value = 1
            elif state[2*i:2*i + 2] == (1,0):
                self.nodes[pos].value = 2
            else:
                raise ValueError("Wrong format")          

    def get_neighbor_pos_with_same_value(self, pos):
        node = self.get_node(pos)
        return [self.positions[neighbor] for neighbor in node.get_all_neighbors_with_same_value()]

    def get_filled_positions(self):
        return [node_pos for node_pos in self.nodes if self.nodes[node_pos].value]

class Diamond(HexGrid):
    """ 
    Diamond grid
    Creates a hex grid of nodes that have correct relations to each other (diamond grid). 
    Overrides methods in hex grid to customize the representation.
    """
    
    def __init__(self,size):
        super(Diamond, self).__init__()
        self.size = size

        for i in range(size):
            for j in range(size):
                self.add_node((i,j))

        for pos in self.get_legal_positions():
            for relative_pos in [(-1,0),(1,0),(0,1),(0,-1),(1,-1),(-1,1)]:
                neighbor_pos = (pos[0]+relative_pos[0],pos[1] + relative_pos[1])
                if neighbor_pos in self.nodes:
                    self.add_edge(pos,neighbor_pos)

    def show_graph(self, debug=False, pause=PAUSE, action_nodes_pos=[]):
        if debug:
            super(Diamond, self).show_graph(debug=True,action_nodes_pos=action_nodes_pos, pause=pause)
        else: 
            positions = {}
            for pos in self.get_legal_positions():
                x = pos[1]-pos[0]
                y = 2*self.size - pos[0] - pos[1]
                positions[pos]=(x,y)
            super(Diamond, self).show_graph(positions=positions, pause=pause, action_nodes_pos=action_nodes_pos)   
