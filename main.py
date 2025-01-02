# So files can be run independently or as a submodule
try:
    from . import config
except ImportError:
    import config

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from typing import List


top_node_ids = []
bottom_node_ids = []
left_node_ids = []
right_node_ids = []

player_node_id = f'{config.NUM_SCREEN_ROWS // 2}_{config.NUM_SCREEN_COLS // 2}'

def init_node_id_lists():
    global top_node_ids
    global bottom_node_ids
    global left_node_ids
    global right_node_ids

    for i in range(config.NUM_SCREEN_COLS):
        top_node_ids.append(f'0_{i}')
        bottom_node_ids.append(f'{config.NUM_SCREEN_ROWS - 1}_{i}')
    
    for i in range(config.NUM_SCREEN_ROWS):
        left_node_ids.append(f'{i}_0')
        right_node_ids.append(f'{i}_{config.NUM_SCREEN_COLS - 1}')


def create_graph_from_state(state: List[List[int]]) -> nx.DiGraph:
    """Creates a graph for a given environment state

    Args:
      state: 2D array of integers
    
    Returns:
      Directional graph representing the given state
    """
    G = nx.DiGraph()

    # Add nodes
    for i in range(config.NUM_SCREEN_ROWS):
        for j in range(config.NUM_SCREEN_COLS):
            node_id = f'{i}_{j}'
            G.add_node(node_id, pos=(j, -i), value=state[i][j])

    # Add edges
    edges_to_add = []
    for i in range(config.NUM_SCREEN_ROWS):
        for j in range(config.NUM_SCREEN_COLS):
            node_id = f"{i}_{j}"
            node_edges = connect_node(G, node_id)
            if len(node_edges) > 0:
                edges_to_add.extend(node_edges)
    
    G.add_edges_from(edges_to_add)
    
    return G


def draw_graph(graph):
    pos = nx.get_node_attributes(graph, 'pos')
    node_values = [node[1]['value'] for node in graph.nodes(data=True)]
    node_colors = [config.COLOR_MAP[value] for value in node_values]

    nx.draw(graph, pos, node_color=node_colors, with_labels=True)
    plt.show()


def get_shortest_path(graph, start, end):
    try:
        path = nx.shortest_path(graph, source=start, target=end)
        return path
    except nx.NetworkXNoPath:
        return None


def get_commands(path):
    commands = []
    current = path[0].split('_')
    current_row, current_col = int(current[0]), int(current[1])
    for i in range(1, len(path)):
        next = path[i].split('_')
        next_row, next_col = int(next[0]), int(next[1])
        if current_row != next_row:
            # needs to move up or down
            cmd = 'up' if current_row > next_row else 'down'
        elif current_col != next_col:
            # needs to move left or right
            cmd = 'left' if current_col > next_col else 'right'
        commands.append(cmd)

        current_row, current_col = next_row, next_col
    return commands


def connect_node(graph, node_id) -> list:
    '''Grabs the id\'s of its neighbors and appropriately connects to each one'''
    if graph.nodes[node_id]['value'] in (0, 2):
        return []
    
    edges = []
    row, col = node_id.split('_')
    row = int(row)
    col = int(col)
    nodes = graph.nodes

    # Check if node exists above
    above_id = f'{row - 1}_{col}'
    if above_id in graph:
        # Node can connect to above node if above node is open space, entrance/exit, or grass (13, 6)
        # And current node is not ledge (4)
        if nodes[above_id]['value'] in (1, 3, 6) and nodes[node_id]['value'] != 4:
            edges.append((node_id, above_id))

    # Check if node exists below
    below_id = f'{row + 1}_{col}'
    if below_id in graph:
        # Node can connect to below node if below node is open space, entrance/exit, ledge, or grass (1, 3, 4, 6)
        if nodes[below_id]['value'] in (1, 3, 4, 6):
            edges.append((node_id, below_id))

    # Check if node exists left
    left_id = f'{row}_{col - 1}'
    if left_id in graph:
        # Node can connect to left node if left node is open space, entrance/exit, grass (1, 3, 6)
        # And current node is not ledge (4)
        if nodes[left_id]['value'] in (1, 3, 6) and nodes[node_id]['value'] != 4:
            edges.append((node_id, left_id))

    # Check if node exists right
    right_id = f'{row}_{col + 1}'
    if right_id in graph:
        # Node can connect to right node if right node is open space, entrance/exit, grass (1, 3, 6)
        # And current node is not ledge (4)
        if nodes[right_id]['value'] in (1, 3, 6) and nodes[node_id]['value'] != 4:
            edges.append((node_id, right_id))
    
    return edges


def get_node_id(row: int, col: int) -> str:
    """Calculates a node's ID based on given row and column on screen

    Args:
      row: Row of node in current screen
      col: Column of node in current screen

    Returns:
      Node ID
    """
    global top_node_ids

    # Using node in the top left, should be able to calculate node id by taking that id and adding row and col to it
    # Example row = 4, col = 2 and top left node id is 0_0
    top_left_row, top_left_col = map(int, top_node_ids[0].split('_'))

    node_row = top_left_row + row
    node_col = top_left_col + col

    return f'{node_row}_{node_col}'


def get_shift(arr1, arr2, arr_type):
    # check left shift
    # rows: left shift = right move, right shift = left move
    # cols: left shift = down move, right shift = up move

    # TODO: Update where left and right column both have to have shifted to be considered an up or down move
    # Need to figure out how when comparing the left column, if no moves are made and an NPC steps into
    # the left column, it still needs to be able to determine if a shift has happened or not
    print(arr1)
    print(arr2)
    
    if np.array_equal(arr1, arr2):
        return None
    
    if np.array_equal(arr1[1:], arr2[:-1]):
        return 'right' if arr_type == 'row' else 'down'
    
    if np.array_equal(arr1[:-1], arr2[1:]):
        return 'left' if arr_type == 'row' else 'up'
    
    return None


def get_move_direction(prev_state, new_state):
    '''Returns movement of character based on state change, returns None if no movement was made'''
    top_move = get_shift(prev_state[0], new_state[0], 'row')
    bottom_move = get_shift(prev_state[-1], new_state[-1], 'row')
    left_move = get_shift(prev_state[:, 0], new_state[:, 0], 'col')
    right_move = get_shift(prev_state[:, -1], new_state[:, -1], 'col')

    # Make sure at least one parallel boundary is not None
    if (top_move is not None or bottom_move is not None) and left_move is None and right_move is None:
        return top_move if top_move is not None else bottom_move
    elif (left_move is not None or right_move is not None) and top_move is None and bottom_move is None:
        return left_move if left_move is not None else right_move
    
    return None


def update_player_loc_on_graph(graph, state, move):
    global player_node_id

    prev_row_idx = config.NUM_SCREEN_ROWS // 2
    prev_col_idx = config.NUM_SCREEN_COLS // 2

    row, col = map(int, player_node_id.split('_'))
    prev_node_id = f'{row}_{col}'
    if move == 'up':
        player_node_id = f'{row - 1}_{col}'
        prev_row_idx += 1
    elif move == 'down':
        player_node_id = f'{row + 1}_{col}'
        prev_row_idx -= 1
    elif move == 'left':
        player_node_id = f'{row}_{col - 1}'
        prev_col_idx += 1
    elif move == 'right':
        player_node_id = f'{row}_{col + 1}'
        prev_col_idx -= 1
    
    # Calculate index for tile that player just moved from to get its value
    graph.nodes[prev_node_id]['value'] = state[prev_row_idx][prev_col_idx]
    graph.nodes[player_node_id]['value'] = 5

    # Remove edges from both nodes to update with new state value
    prev_neighbors = set(graph.successors(prev_node_id)).union(set(graph.predecessors(prev_node_id)))
    player_neighbors = set(graph.successors(player_node_id)).union(set(graph.predecessors(player_node_id)))
    edges_to_remove = list(graph.in_edges(prev_node_id)) + list(graph.out_edges(prev_node_id))
    edges_to_remove.extend(list(graph.in_edges(player_node_id)) + list(graph.out_edges(player_node_id)))
    graph.remove_edges_from(edges_to_remove)

    # Connect both nodes correctly with their neighbors based on new state value
    edges = []
    for node in prev_neighbors:
        edges.extend(connect_node(graph, node))
    for node in player_neighbors:
        edges.extend(connect_node(graph, node))
    graph.add_edges_from(edges)

    return graph


def update_screen_boundaries(new_node_ids, move):
    global top_node_ids
    global bottom_node_ids
    global left_node_ids
    global right_node_ids

    if move == 'up':
        # Update top_node_ids with list of new node IDs
        top_node_ids = new_node_ids[:]

        # left_node_ids: top_node_ids[0] is added to front of list, pop last item from left_node_ids
        left_node_ids.insert(0, top_node_ids[0])
        left_node_ids.pop()

        # right_node_ids: top_node_ids[-1] is added to front of list, pop last item from right_node_ids
        right_node_ids.insert(0, top_node_ids[-1])
        right_node_ids.pop()

        # bottom_node_ids: iterate through and decrement each row aka {row - 1}_{col}
        for i in range(config.NUM_SCREEN_COLS):
            row, col = map(int, bottom_node_ids[i].split('_'))
            bottom_node_ids[i] = f'{row - 1}_{col}'

    elif move == 'down':
        # Update bottom_node_ids with list of new node IDs
        bottom_node_ids = new_node_ids[:]

        # left_node_ids: bottom_node_ids[0] is added to the end of list, remove first item from left_node_ids
        left_node_ids.append(bottom_node_ids[0])
        left_node_ids.pop(0)

        # right_node_ids: bottom_node_ids[-1] is added to front of list, pop last item from right_node_ids
        left_node_ids.append(bottom_node_ids[-1])
        left_node_ids.pop(0)

        # top_node_ids: iterate through and increment each row aka {row + 1}_{col}
        for i in range(config.NUM_SCREEN_COLS):
            row, col = map(int, top_node_ids[i].split('_'))
            top_node_ids[i] = f'{row + 1}_{col}'

    elif move == 'left':
        # Update left_node_ids with list of new node IDs
        left_node_ids = new_node_ids[:]

        # top_node_ids: left_node_ids[0] is added to the front of list, remove last item from top_node_ids
        top_node_ids.insert(0, left_node_ids[0])
        top_node_ids.pop()        

        # bottom_node_ids: left_node_ids[-1] is added to the front of list, remove last item from bottom_node_ids
        bottom_node_ids.insert(0, left_node_ids[-1])
        bottom_node_ids.pop()

        # right_node_ids: iterate through and decrement each row aka {row}_{col - 1}
        for i in range(config.NUM_SCREEN_ROWS):
            row, col = map(int, right_node_ids[i].split('_'))
            right_node_ids[i] = f'{row}_{col - 1}'
        
    elif move == 'right':
        # Update right_node_ids with list of new node IDs
        right_node_ids = new_node_ids[:]

        # top_node_ids: right_node_ids[0] is added to the end of list, remove first item from top_node_ids
        top_node_ids.append(right_node_ids[0])
        top_node_ids.pop(0)

        # bottom_node_ids: right_node_ids[-1] is added to the end of list, remove first item from bottom_node_ids
        bottom_node_ids.append(right_node_ids[-1])
        bottom_node_ids.pop(0)

        # left_node_ids: iterate through and increment each row aka {row}_{col + 1}
        for i in range(config.NUM_SCREEN_ROWS):
            row, col = map(int, left_node_ids[i].split('_'))
            left_node_ids[i] = f'{row}_{col + 1}'


def find_state_differences(prev_state, new_state):
    '''Returns list of indices where value changed with old and new value'''
    prev_state = np.array(prev_state)
    new_state = np.array(new_state)

    diff_indices = np.argwhere(prev_state != new_state)

    differences = []
    for index in diff_indices:
        row, col = map(int, index)
        node_id = get_node_id(row, col)
        old_value = int(prev_state[row, col])
        new_value = int(new_state[row, col])
        # Ignore difference if it's the player moving (x -> 5 or 5 -> x)
        if old_value != 5 and new_value != 5:
            differences.append((node_id, (old_value, new_value)))
    
    return differences


def update_graph(graph, prev_state, new_state):
    global top_node_ids
    global bottom_node_ids
    global left_node_ids
    global right_node_ids
    
    move = get_move_direction(prev_state, new_state)
    if move is None:
        print('Player did not move')
        # find any differences between existing node values and values in new_state
        differences = find_state_differences(prev_state, new_state)
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
                edges.extend(connect_node(graph, node))
            graph.add_edges_from(edges)

    else:
        print(f'Player moved {move}')
        # Assign variables with the correct value based on move direction
        if move in ('up', 'down'):
            num_new_nodes = config.NUM_SCREEN_COLS
            edge_row_node_ids = top_node_ids if move == 'up' else bottom_node_ids
            edge_row_num = int(edge_row_node_ids[0].split('_')[0])
            new_row_num = edge_row_num - 1 if move == 'up' else edge_row_num + 1
            new_env = new_state[0] if move == 'up' else new_state[-1]
        elif move in ('left', 'right'):
            num_new_nodes = config.NUM_SCREEN_ROWS
            edge_row_node_ids = left_node_ids if move == 'left' else right_node_ids
            edge_row_num = int(edge_row_node_ids[0].split('_')[1])
            new_row_num = edge_row_num - 1 if move == 'left' else edge_row_num + 1
            new_env = [row[0] for row in new_state] if move == 'left' else [row[-1] for row in new_state]

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
            edges_to_remove = list(graph.in_edges(node_id)) + list(graph.out_edges(node_id))
            graph.remove_edges_from(edges_to_remove)
        graph.add_nodes_from(new_nodes)

        edges_to_add = []
        for i in range(num_new_nodes):
            edge_edges = connect_node(graph, edge_row_node_ids[i])
            new_edges = connect_node(graph, new_node_ids[i])
            new_edges.extend(edge_edges)
            if len(new_edges) > 0:
                edges_to_add.extend(new_edges)
        graph.add_edges_from(edges_to_add)

        # Get any differences in values between overlapping parts of states
        if move == 'up':
            # all of new_state except for first row
            # all of prev_state except for last row
            differences = find_state_differences(prev_state[:-1], new_state[1:])
        elif move == 'down':
            # all of prev_state except first row
            # all of new_state except last row
            differences = find_state_differences(prev_state[1:], new_state[:-1])
        elif move == 'left':
            # all of prev_state except last column
            # all of new_state except first column
            differences = find_state_differences(prev_state[:, :-1], new_state[:, 1:])
        elif move == 'right':
            # all of prev_state except first column
            # all of new_state except last column
            differences = find_state_differences(prev_state[:, 1:], new_state[:, :-1])
        
        # Update graph from differences
        print('Differences:', differences)
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
                edges.extend(connect_node(graph, node))
            graph.add_edges_from(edges)

        # Update player location on graph
        graph = update_player_loc_on_graph(graph, new_state, move)

        # Update screen boundary lists
        update_screen_boundaries(new_node_ids, move)

    return graph


prev_state = None
states = [
    [
        [0, 1, 1, 1, 1, 1, 1, 1, 6, 6, 6, 6, 6, 6, 1],
        [2, 1, 1, 1, 1, 4, 4, 4, 4, 6, 6, 6, 6, 6, 1],
        [6, 6, 1, 1, 1, 1, 1, 1, 1, 0, 0, 6, 6, 1, 1],
        [6, 6, 6, 1, 1, 1, 1, 1, 1, 0, 0, 6, 6, 1, 1],
        [6, 6, 6, 1, 1, 1, 1, 5, 1, 0, 0, 0, 0, 0, 0],
        [6, 6, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
    ],
    [
        # move down
        [2, 1, 1, 1, 1, 4, 4, 4, 4, 6, 6, 6, 6, 6, 1],
        [6, 6, 1, 1, 1, 1, 1, 1, 1, 0, 0, 6, 6, 1, 1],
        [6, 6, 6, 1, 1, 1, 1, 1, 1, 0, 0, 6, 6, 1, 1],
        [6, 6, 6, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [6, 6, 1, 1, 1, 1, 1, 5, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
    ],
    [
        # move down
        [6, 6, 1, 1, 1, 1, 1, 1, 1, 0, 0, 6, 6, 1, 1],
        [6, 6, 6, 1, 1, 1, 1, 1, 1, 0, 0, 6, 6, 1, 1],
        [6, 6, 6, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [6, 6, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 5, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ],
    [
        # move down
        [6, 6, 6, 1, 1, 1, 1, 1, 1, 0, 0, 6, 6, 1, 1],
        [6, 6, 6, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [6, 6, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 5, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ],
    [
        # move up
        [6, 6, 1, 1, 1, 1, 1, 1, 1, 0, 0, 6, 6, 1, 1],
        [6, 6, 6, 1, 1, 1, 1, 1, 1, 0, 0, 6, 6, 1, 1],
        [6, 6, 6, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [6, 6, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 5, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ],
    [
        # move up
        [2, 1, 1, 1, 1, 4, 4, 4, 4, 6, 6, 6, 6, 6, 1],
        [6, 6, 1, 1, 1, 1, 1, 1, 1, 0, 0, 6, 6, 1, 1],
        [6, 6, 6, 1, 1, 1, 1, 1, 1, 0, 0, 6, 6, 1, 1],
        [6, 6, 6, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [6, 6, 1, 1, 1, 1, 1, 5, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
    ],
    [
        # move up
        [0, 1, 1, 1, 1, 1, 1, 1, 6, 6, 6, 6, 6, 6, 1],
        [2, 1, 1, 1, 1, 4, 4, 4, 4, 6, 6, 6, 6, 6, 1],
        [6, 6, 1, 1, 1, 1, 1, 1, 1, 0, 0, 6, 6, 1, 1],
        [6, 6, 6, 1, 1, 1, 1, 1, 1, 0, 0, 6, 6, 1, 1],
        [6, 6, 6, 1, 1, 1, 1, 5, 1, 0, 0, 0, 0, 0, 0],
        [6, 6, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
    ]
]

for state in states:
    if prev_state is None:
        init_node_id_lists()
        graph = create_graph_from_state(state)
    else:
        prev_state = np.array(prev_state)
        state = np.array(state)
        graph = update_graph(graph, prev_state, state)
    
    prev_state = state[:]
    draw_graph(graph)
