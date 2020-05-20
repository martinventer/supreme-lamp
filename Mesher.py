# Mesher

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import RegularPolygon

import networkx as nx
import itertools

from collections import defaultdict, Counter


def make_hex_grid(dimension) -> dict:
    """
    Create a dictionary of hexagonal grid elements.
    :param dimension: int
        radial dimension for the regular hex grid
    :return: dict
        {element_ID: [x-coord, y-coord, z-coord], }
    """
    coord_list = list()
    for x in range(-1 * dimension, dimension + 1, 1):
        for y in range(-1 * dimension, dimension + 1, 1):
            for z in range(-1 * dimension, dimension + 1, 1):
                if sum([x, y, z]) is 0:
                    coord_list.append([x, y, z])

    return {element_id: coord for element_id, coord in enumerate(coord_list)}


def hex_to_cart(hex_coord) -> (list, list):
    """
    Takes a list of hex coord and converts them to a list of cartesian
    coordinates.
    :param hex_coord: list
        list of hex coordinates [[x, y, z], ]
    :return: (list, list)
        pair of coordinate lists x and y ([x], [y])
    """
    x_coord = [c[0] for c in hex_coord]
    y_coord = [2. * np.sin(np.radians(60)) * (c[1] - c[2]) / 3. for c in
               hex_coord]

    return x_coord, y_coord


def plot_hex(elements, active=None) -> None:
    """
    Plots a hex_patch grid with color and label overlay. Conversion of
    hexagon to cartesian coordinates.
    :param elements: dict
        dictionary of hex_patch elements
    :param active: dict
        dictionary of active elements, must be same length as elements
    :return: None
    """
    # handle colours
    if not active:
        color_map = ["green"] * len(elements)
    else:
        color_map = list()
        for element in elements.keys():
            if element in active:
                color_map.append('blue')
            else:
                color_map.append('red')

    # create label and coord  list
    labels = elements.keys()
    hex_coord = elements.values()
    # convert from hex_patch [x, y, z] to cart [x, y]
    x_coord, y_coord = hex_to_cart(hex_coord)

    # process the figure
    fig, ax = plt.subplots(1, figsize=(12, 12))
    ax.set_aspect('equal')

    for x, y, c, l in zip(x_coord, y_coord, color_map, labels):
        hex_patch = RegularPolygon((x, y),
                                   numVertices=6,
                                   radius=2. / 3.,
                                   orientation=np.radians(30),
                                   facecolor=c,
                                   alpha=0.2,
                                   edgecolor='k')

        ax.add_patch(hex_patch)
        # Also add a text label
        ax.text(x, y + 0.2, l, ha='center', va='center', size=20)

    # Also add scatter points in hexagon centres
    ax.scatter(x_coord, y_coord, c=color_map, alpha=0.5)
    plt.show()
    return None


def active_subset_random(network, depth=3) -> list:
    """
    Creates a dictionary of random active assignments.
    :param depth: int
        the number of active elements to return
    :param network: object
        Networkx graph
    :return: dict
        a dictionary containing the element id and a boolean indicator of
        activity
    """
    node_list = list(network.nodes)
    return np.random.choice(node_list, size=depth, replace=False).tolist()


def unique_element_pairs(elements) -> tuple:
    """
    Generates a list of unique element pairs for the grid.
    :param elements: dict
        dictionary containing element ids and hex coordinates
    :return: tuple
        generates the next unique element pair in the grid
    """
    # keep a set of pairs that have already been seen
    seen = set()
    possible_pairs = list(itertools.combinations(elements.keys(), 2))
    # iterate over each pair and test for uniqueness
    for pair in possible_pairs:
        # test whether that pair has been seen before
        if pair not in seen:
            # if not seen before append the pair and its reverse to seen
            seen.add(pair[::-1])
            seen.add(pair)
            yield pair


def test_element_adjacency(elements, pair) -> bool:
    """
    test whether a pair of elements are adjacent to each other
    :param elements: dict
        dictionary containing element ids and hex coordinates
    :param pair: tuple
        pair of element ids
    :return: bool
        True if elements are adjacent of False if they are not.
    """
    element_a = elements[pair[0]]
    element_b = elements[pair[1]]
    difference = [a - b for a, b in zip(element_a, element_b)]
    # elements are adjacent if at lease one of their hex coordinates differ by 1
    if ((np.abs(difference[0]) <= 1) and
            (np.abs(difference[1]) <= 1) and
            (np.abs(difference[2]) <= 1)):
        return True
    else:
        return False


def make_hex_network(elements) -> object:
    """
    Test each pair of elements in a grid and add an edge for each pair
    :param elements: dict
        dictionary containing element ids and hex coordinates
    :return: object
        returns a network object
    """
    pair_gen = unique_element_pairs(elements)

    attrs = dict()
    for elid, coords in elements.items():
        attrs[elid] = {'hex_coord': coords}

    graph = nx.Graph()
    for edge in pair_gen:
        if test_element_adjacency(elements, edge):
            graph.add_edge(*edge)

    nx.set_node_attributes(graph, attrs)

    return graph


def plot_network(network, active=None) -> None:
    """
    Plot the network with the subset highlighted.
    :param active: list
        list of active elements, must be same length as elements
    :param network: object
        NetworkX network object
    :return: object
        The network with updated colours
    """
    # handle colours
    if not active:
        color_map = ["green"] * len(network)
    else:
        color_map = list()
        for node in network:
            if node in active:
                color_map.append('blue')
            else:
                color_map.append('red')

    # process the figure
    fig, ax = plt.subplots(1, figsize=(12, 12))
    ax.set_aspect('equal')
    pos = nx.spring_layout(network)
    nx.draw(network, pos=pos, node_color=color_map, with_labels=True)
    plt.show()
    return None


def get_active_sub_network(network, active=None) -> object:
    """
    generates a subnetwork the network with the subset highlighted.
    :param active: dict
        dictionary of active elements, must be same length as elements
    :param network: object
        NetworkX network object
    :return: object
        The sub-network
    """
    return network.subgraph(active)


def test_network_connectivity(network) -> bool:
    """
    Test a network to see if it is fully connected. (wrapper for
    nx.is_connected())
    :param network: object
        NetworkX network object
    :return: bool
        Returns True if the network is connected else False
    """
    if not network:
        return False
    else:
        return nx.is_connected(network)


def test_network_isomorphs(network_1, network_2) -> bool:
    """
    Test whether two networks are isomorphous. (Wrapper for nx.is_isomorphic())
    :param network_1: object
        NetworkX network object
    :param network_2: object
        NetworkX network object
    :return: bool
        Returns True if two networks are isomorphs else False
    """
    if not network_1 or not network_2:
        return False
    else:
        return nx.is_isomorphic(network_1, network_2)


def active_subset_markov_random(network, depth=3, starting_node=None) -> list:
    """
    From a given starting point create branching dictionary. Kind of a greedy
    approach, but what you gone do?
    :param network: object
        Networkx graph
    :param starting_node: int, string
        name of the starting node. Default is None, in which case the most
        central is chosen
    :param depth: int
        The number of active nodes to be returned
    :return:
        generates
    """
    if not starting_node:
        starting_node = nx.center(network)[0]

    active_list = [starting_node]
    current_node = starting_node
    for element in range(depth-1):
        adjacent_nodes = [n for n in network.neighbors(current_node)]
        adjacent_nodes = [x for x in adjacent_nodes if x not in active_list]
        if not adjacent_nodes:
            break
        next_node = np.random.choice(adjacent_nodes, size=1).tolist()[0]
        active_list.append(next_node)
        current_node = next_node

    return active_list


def find_all_isomorphs(network, depth=3, runs=100) -> dict:
    """
    Generate a number of active subsets, evaluate whether they exist as
    isomprphs in the dict.
    :param network: object
        Networkx graph
    :param depth: int
        The number of active nodes to be returned
    :param runs: int
        The number of runs to do.
    :return: dict
        Returns a dictionary containing
    """
    isomprphs = dict()
    isomprphs[0] = [active_subset_markov_random(network, depth=depth)]
    counter = 1
    for _ in range(runs):
        new_morph_flag = True
        # Generate a new random active set, and sub network
        active_subset = active_subset_markov_random(network, depth=depth)
        new_subnet = get_active_sub_network(network, active=active_subset)
        # compare the new subnet to the first item in each isomorph
        for idx, isomprph in isomprphs.items():
            target_subnet = get_active_sub_network(network, active=isomprph[0])
            # if an isomorph exists append this version to that isomprph
            if test_network_isomorphs(new_subnet, target_subnet):
                # check for duplicates in the isomorph list
                duplicate_morph_flag = False
                for morph in isomprph:
                    if set(active_subset) == set(morph):
                        new_morph_flag = False
                        duplicate_morph_flag = True
                        break

                if not duplicate_morph_flag:
                    isomprphs[idx].append(active_subset)
                    new_morph_flag = False
        # if an isomorph does not exist create an new dictionary item for it.
        if new_morph_flag:
            isomprphs[counter] = [active_subset]
            counter += 1

    return isomprphs


def test_compare_two_lists(list1, list2) -> bool:
    """
    test whether two lists contain the same items
    :param list1: list
    :param list2: list
    :return: bool
        True if lists are the same otherwise false.
    """
    return Counter(list1) == Counter(list2)


def hex_rotate_60(coord_list) -> list:
    """
    Rotates a hex co-ordinate clockwise 60 degrees
    :param coord_list: list
        hex co-ordinate list [x, y, z]
    :return: list
        Rotated hex co-ordinate list [x, y, z]
    """
    coord_list.insert(0, coord_list.pop())
    return [-x for x in coord_list]


def eid_lookup(hexmesh) -> dict:
    """
    creates a lookup table for element ID in a mesh based on their coordinates
    :param hexmesh: dict
        a hex mesh dictionary
    :return: dict
        Returns dictionary returning the eid in form dict[x][y][z]
    """
    # create a defaultdict builder with infinate depth
    factory = lambda: defaultdict(factory)

    lookup = defaultdict(factory)
    for eid, coord in hexmesh.items():
        x, y, z = coord
        lookup[x][y][z] = eid

    return lookup


def rotate_subset(hexmesh, subset) -> list:
    """
    Takes a subset and mesh, returns the rotated set
    :param hexmesh: dict
        Dictionary containg the full mesh
    :param subset:  list
        list of elements in the subset
    :return: list
        list of rotated elements
    """
    el_map = eid_lookup(hexmesh)
    rotated_list = list()
    for element in subset:
        coord = hexmesh[element].copy()
        rotated_coord = hex_rotate_60(coord)
        x, y, z = rotated_coord
        rotated_list.append(el_map[x][y][z])

    return rotated_list


if __name__ == '__main__':
    print("Start Mesher.py")

    print("End Mesher.py")
