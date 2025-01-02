# Create a class that an object can be made from in ARES project
# so that you can do something like nexus.add_state(new_state)
# and it will add to or update the graph and assign the current
# state to prev_state and the new state to the current state

# So files can be run independently or as a submodule
try:
    from . import config
except ImportError:
    import config

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


class Graph:
    def __init__(self):
        # Keeps track of the node IDs on the sides of the current screen
        self.top_node_ids = []
        self.bottom_node_ids = []
        self.left_node_ids = []
        self.right_node_ids = []

        self.graph = nx.DiGraph()

        self.prev_state: np.ndarray = None
        self.current_state: np.ndarray = None

        self.player_node_id = f'{config.NUM_SCREEN_ROWS // 2}_{config.NUM_SCREEN_COLS // 2}'


    # Public methods
    def update(self, state):
        state = np.array(state)
        if self.current_state is None:
            self.__create_graph_from_state(state)
            return
        
        self.prev_state = self.current_state[:]
        self.current_state = state

        move = self.__get_move_direction()
        # Player didn't move
        if move is None:
            # Get any differences between current state and previous state
            differences = self.__find_state_differences(self.prev_state, self.current_state)

            for node_id, (_, new_value) in differences:
                self.graph.nodes[node_id]['value'] = new_value

                # Get all of node's neighbors
                neighbors = set(self.graph.successors(node_id)).union(set(self.graph.predecessors(node_id)))

                # Remove all in out edges for node
                edges_to_remove = list(self.graph.in_edges(node_id)) + list(self.graph.out_edges(node_id))
                self.graph.remove_edges_from(edges_to_remove)

                # Connect node with its neighbors accordingly
                edges = []
                for node in neighbors:
                    edges.extend(self.__connect_node(node))
                self.graph.add_edges_from(edges)

        # Player moved
        else:
            if move in ('up', 'down'):
                num_new_nodes = config.NUM_SCREEN_COLS
                edge_row_node_ids = self.top_node_ids if move == 'up' else self.bottom_node_ids
                edge_row_num = int(edge_row_node_ids[0].split('_')[0])
                new_row_num = edge_row_num - 1 if move == 'up' else edge_row_num + 1
                new_env = self.current_state[0] if move == 'up' else self.current_state[-1]
            elif move in ('left', 'right'):
                num_new_nodes = config.NUM_SCREEN_ROWS
                edge_row_node_ids = self.left_node_ids if move == 'left' else self.right_node_ids
                edge_row_num = int(edge_row_node_ids[0].split('_')[1])
                new_row_num = edge_row_num - 1 if move == 'left' else edge_row_num + 1
                new_env = [row[0] for row in self.current_state] if move == 'left' else [row[-1] for row in self.current_state]

            # Need to figure out how to not touch nodes that already exist in graph
            # if moving back to an area that has already been discovered
            # while still being able to change a node if it's value is different when you're back in that area
            # moved up and the row right before the ledge got added again which removed the down edge from the ledge
            # the edges got added back in, once the ledge was at the top of the screen the down edges from it got added back
            # once the row above the ledge was shown, the edge into the ledge was added back
            # maybe it isn't an issue?

            # Create nodes in the graph for the new points
            new_node_ids = []
            new_nodes = []
            for i in range(len(edge_row_node_ids)):
                row, col = map(int, edge_row_node_ids[i].split('_'))
                if move in ('up', 'down'):
                    node_id = f'{new_row_num}_{col}'
                    pos = (col, -new_row_num)
                elif move in ('left', 'right'):
                    node_id = f'{row}_{new_row_num}'
                    pos = (new_row_num, -row)
                new_node_ids.append(node_id)
                new_nodes.append((node_id, {'pos': pos, 'value': new_env[i]}))
                edges_to_remove = list(self.graph.in_edges(node_id)) + list(self.graph.out_edges(node_id))
                self.graph.remove_edges_from(edges_to_remove)
            self.graph.add_nodes_from(new_nodes)

            edges_to_add = []
            for i in range(num_new_nodes):
                edge_edges = self.__connect_node(edge_row_node_ids[i])
                new_edges = self.__connect_node(new_node_ids[i])
                new_edges.extend(edge_edges)
                if len(new_edges) > 0:
                    edges_to_add.extend(new_edges)
            self.graph.add_edges_from(edges_to_add)

            # Get any differences in values between overlapping parts of states
            if move == 'up':
                # all of new_state except for first row
                # all of prev_state except for last row
                differences = self.__find_state_differences(self.prev_state[:-1], self.current_state[1:])
            elif move == 'down':
                # all of prev_state except first row
                # all of new_state except last row
                differences = self.__find_state_differences(self.prev_state[1:], self.current_state[:-1])
            elif move == 'left':
                # all of prev_state except last column
                # all of new_state except first column
                differences = self.__find_state_differences(self.prev_state[:, :-1], self.current_state[:, 1:])
            elif move == 'right':
                # all of prev_state except first column
                # all of new_state except last column
                differences = self.__find_state_differences(self.prev_state[:, 1:], self.current_state[:, :-1])
            
            # Update graph from differences
            for node_id, (_, new_value) in differences:
                # Update node with its new value
                graph.nodes[node_id]['value'] = new_value

                # Get all of node's neighbors
                neighbors = set(graph.successors(node_id)).union(set(graph.predecessors(node_id)))

                # Remove all in out edges for node
                edges_to_remove = list(graph.in_edges(node_id)) + list(graph.out_edges(node_id))
                graph.remove_edges_from(edges_to_remove)

                # Connect node with its neighbors accordingly
                edges = []
                for node in neighbors:
                    edges.extend(self.__connect_node(node))
                graph.add_edges_from(edges)

            # Update player location on graph
            graph = self.__update_player_loc_on_graph(move)

            # Update screen boundary lists
            self.__update_screen_boundaries(new_node_ids, move)        

    def display(self):
        pos = nx.get_node_attributes(self.graph, 'pos')
        node_values = [node[1]['value'] for node in self.graph.nodes(data=True)]
        node_colors = [config.COLOR_MAP[value] for value in node_values]

        nx.draw(self.graph, pos, node_color=node_colors, with_labels=True)
        plt.show()

    def save_graph(self):
        # saves graph to file that can be loaded
        pass

    def load_graph(self):
        pass


    # Private methods
    def __init_node_id_lists(self):
        for i in range(config.NUM_SCREEN_COLS):
            self.top_node_ids.append(f'0_{i}')
            self.bottom_node_ids.append(f'{config.NUM_SCREEN_ROWS - 1}_{i}')
        
        for i in range(config.NUM_SCREEN_ROWS):
            self.left_node_ids.append(f'{i}_0')
            self.right_node_ids.append(f'{i}_{config.NUM_SCREEN_COLS - 1}')

    def __create_graph_from_state(self, state: np.ndarray):
        self.__init_node_id_lists()

        # Add nodes
        for i in range(config.NUM_SCREEN_ROWS):
            for j in range(config.NUM_SCREEN_COLS):
                node_id = f'{i}_{j}'
                self.graph.add_node(node_id, pos=(j, -i), value=state[i][j])

        # Add edges
        edges_to_add = []
        for i in range(config.NUM_SCREEN_ROWS):
            for j in range(config.NUM_SCREEN_COLS):
                node_id = f"{i}_{j}"
                node_edges = self.__connect_node(node_id)
                if len(node_edges) > 0:
                    edges_to_add.extend(node_edges)
        
        self.graph.add_edges_from(edges_to_add)

        self.current_state = np.array(state)

    def __connect_node(self, node_id: str) -> list:
        """Grabs the IDs of its neighbors and appropriately connects to each one"""
        if self.graph.nodes[node_id]['value'] in (0, 2):
            return []
        
        edges = []
        row, col = node_id.split('_')
        row = int(row)
        col = int(col)
        nodes = self.graph.nodes

        # Check if node exists above
        above_id = f'{row - 1}_{col}'
        if above_id in self.graph:
            # Node can connect to above node if above node is open space, entrance/exit, or grass (13, 6)
            # And current node is not ledge (4)
            if nodes[above_id]['value'] in (1, 3, 6) and nodes[node_id]['value'] != 4:
                edges.append((node_id, above_id))

        # Check if node exists below
        below_id = f'{row + 1}_{col}'
        if below_id in self.graph:
            # Node can connect to below node if below node is open space, entrance/exit, ledge, or grass (1, 3, 4, 6)
            if nodes[below_id]['value'] in (1, 3, 4, 6):
                edges.append((node_id, below_id))

        # Check if node exists left
        left_id = f'{row}_{col - 1}'
        if left_id in self.graph:
            # Node can connect to left node if left node is open space, entrance/exit, grass (1, 3, 6)
            # And current node is not ledge (4)
            if nodes[left_id]['value'] in (1, 3, 6) and nodes[node_id]['value'] != 4:
                edges.append((node_id, left_id))

        # Check if node exists right
        right_id = f'{row}_{col + 1}'
        if right_id in self.graph:
            # Node can connect to right node if right node is open space, entrance/exit, grass (1, 3, 6)
            # And current node is not ledge (4)
            if nodes[right_id]['value'] in (1, 3, 6) and nodes[node_id]['value'] != 4:
                edges.append((node_id, right_id))
        
        return edges

    def __get_shift(self, arr1, arr2, arr_type):
        # check left shift
        # rows: left shift = right move, right shift = left move
        # cols: left shift = down move, right shift = up move

        # TODO: Update where left and right column both have to have shifted to be considered an up or down move
        # Need to figure out how when comparing the left column, if no moves are made and an NPC steps into
        # the left column, it still needs to be able to determine if a shift has happened or not
        if np.array_equal(arr1, arr2):
            return None
        
        if np.array_equal(arr1[1:], arr2[:-1]):
            return 'right' if arr_type == 'row' else 'down'
        
        if np.array_equal(arr1[:-1], arr2[1:]):
            return 'left' if arr_type == 'row' else 'up'
        
        return None

    def __get_move_direction(self):
        """Returns movement of character based on state change, returns None if no movement was made"""
        top_move = self.__get_shift(self.prev_state[0], self.current_state[0], 'row')
        bottom_move = self.__get_shift(self.prev_state[-1], self.current_state[-1], 'row')
        left_move = self.__get_shift(self.prev_state[:, 0], self.current_state[:, 0], 'col')
        right_move = self.__get_shift(self.prev_state[:, -1], self.current_state[:, -1], 'col')

        # Make sure at least one parallel boundary is not None
        if (top_move is not None or bottom_move is not None) and left_move is None and right_move is None:
            return top_move if top_move is not None else bottom_move
        elif (left_move is not None or right_move is not None) and top_move is None and bottom_move is None:
            return left_move if left_move is not None else right_move
        
        return None

    def __find_state_differences(self, prev_state, new_state):
        """Returns list of indices where value changed with old and new value"""
        diff_indices = np.argwhere(prev_state != new_state)

        differences = []
        for index in diff_indices:
            row, col = map(int, index)
            node_id = self.__get_node_id(row, col)
            old_value = int(prev_state[row, col])
            new_value = int(new_state[row, col])
            # Ignore difference if it's the player moving (x -> 5 or 5 -> x)
            if old_value != 5 and new_value != 5:
                differences.append((node_id, (old_value, new_value)))
        
        return differences

    def __get_node_id(self, row: int, col: int) -> str:
        """Calculates a node's ID based on given row and column on screen

        Args:
        row: Row of node in current screen
        col: Column of node in current screen

        Returns:
        Node ID
        """
        # Using node in the top left, should be able to calculate node id by taking that id and adding row and col to it
        # Example row = 4, col = 2 and top left node id is 0_0
        top_left_row, top_left_col = map(int, self.top_node_ids[0].split('_'))

        node_row = top_left_row + row
        node_col = top_left_col + col

        return f'{node_row}_{node_col}'

    def __update_player_loc_on_graph(self, move):
        prev_row_idx = config.NUM_SCREEN_ROWS // 2
        prev_col_idx = config.NUM_SCREEN_COLS // 2

        row, col = map(int, self.player_node_id.split('_'))
        prev_node_id = f'{row}_{col}'
        if move == 'up':
            self.player_node_id = f'{row - 1}_{col}'
            prev_row_idx += 1
        elif move == 'down':
            self.player_node_id = f'{row + 1}_{col}'
            prev_row_idx -= 1
        elif move == 'left':
            self.player_node_id = f'{row}_{col - 1}'
            prev_col_idx += 1
        elif move == 'right':
            self.player_node_id = f'{row}_{col + 1}'
            prev_col_idx -= 1
        
        # Calculate index for tile that player just moved from to get its value
        self.graph.nodes[prev_node_id]['value'] = self.current_state[prev_row_idx][prev_col_idx]
        self.graph.nodes[self.player_node_id]['value'] = 5

        # Remove edges from both nodes to update with new state value
        prev_neighbors = set(self.graph.successors(prev_node_id)).union(set(self.graph.predecessors(prev_node_id)))
        player_neighbors = set(self.graph.successors(self.player_node_id)).union(set(self.graph.predecessors(self.player_node_id)))
        edges_to_remove = list(self.graph.in_edges(prev_node_id)) + list(self.graph.out_edges(prev_node_id))
        edges_to_remove.extend(list(self.graph.in_edges(self.player_node_id)) + list(self.graph.out_edges(self.player_node_id)))
        self.graph.remove_edges_from(edges_to_remove)

        # Connect both nodes correctly with their neighbors based on new state value
        edges = []
        for node in prev_neighbors:
            edges.extend(self.__connect_node(node))
        for node in player_neighbors:
            edges.extend(self.__connect_node(node))
        self.graph.add_edges_from(edges)

    def __update_screen_boundaries(self, new_node_ids, move):
        if move == 'up':
            # Update top_node_ids with list of new node IDs
            self.top_node_ids = new_node_ids[:]

            # left_node_ids: top_node_ids[0] is added to front of list, pop last item from left_node_ids
            self.left_node_ids.insert(0, self.top_node_ids[0])
            self.left_node_ids.pop()

            # right_node_ids: top_node_ids[-1] is added to front of list, pop last item from right_node_ids
            self.right_node_ids.insert(0, self.top_node_ids[-1])
            self.right_node_ids.pop()

            # bottom_node_ids: iterate through and decrement each row aka {row - 1}_{col}
            for i in range(config.NUM_SCREEN_COLS):
                row, col = map(int, self.bottom_node_ids[i].split('_'))
                self.bottom_node_ids[i] = f'{row - 1}_{col}'

        elif move == 'down':
            # Update bottom_node_ids with list of new node IDs
            self.bottom_node_ids = new_node_ids[:]

            # left_node_ids: bottom_node_ids[0] is added to the end of list, remove first item from left_node_ids
            self.left_node_ids.append(self.bottom_node_ids[0])
            self.left_node_ids.pop(0)

            # right_node_ids: bottom_node_ids[-1] is added to front of list, pop last item from right_node_ids
            self.left_node_ids.append(self.bottom_node_ids[-1])
            self.left_node_ids.pop(0)

            # top_node_ids: iterate through and increment each row aka {row + 1}_{col}
            for i in range(config.NUM_SCREEN_COLS):
                row, col = map(int, self.top_node_ids[i].split('_'))
                self.top_node_ids[i] = f'{row + 1}_{col}'

        elif move == 'left':
            # Update left_node_ids with list of new node IDs
            self.left_node_ids = new_node_ids[:]

            # top_node_ids: left_node_ids[0] is added to the front of list, remove last item from top_node_ids
            self.top_node_ids.insert(0, self.left_node_ids[0])
            self.top_node_ids.pop()

            # bottom_node_ids: left_node_ids[-1] is added to the front of list, remove last item from bottom_node_ids
            self.bottom_node_ids.insert(0, self.left_node_ids[-1])
            self.bottom_node_ids.pop()

            # right_node_ids: iterate through and decrement each row aka {row}_{col - 1}
            for i in range(config.NUM_SCREEN_ROWS):
                row, col = map(int, self.right_node_ids[i].split('_'))
                self.right_node_ids[i] = f'{row}_{col - 1}'
            
        elif move == 'right':
            # Update right_node_ids with list of new node IDs
            self.right_node_ids = new_node_ids[:]

            # top_node_ids: right_node_ids[0] is added to the end of list, remove first item from top_node_ids
            self.top_node_ids.append(self.right_node_ids[0])
            self.top_node_ids.pop(0)

            # bottom_node_ids: right_node_ids[-1] is added to the end of list, remove first item from bottom_node_ids
            self.bottom_node_ids.append(self.right_node_ids[-1])
            self.bottom_node_ids.pop(0)

            # left_node_ids: iterate through and increment each row aka {row}_{col + 1}
            for i in range(config.NUM_SCREEN_ROWS):
                row, col = map(int, self.left_node_ids[i].split('_'))
                self.left_node_ids[i] = f'{row}_{col + 1}'
