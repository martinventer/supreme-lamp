# Mesher

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import RegularPolygon

import networkx as nx
import itertools

import operator as op
from functools import reduce


def make_hex_grid(dimension) -> dict:
    """
    Create a dictionary of hexagonal grid elements.
    :param dimension: int
        radial dimension for the regular hex grid
    :return: dict
        {element_ID: [x-coord, y-coord, z-coord], }
    """
    coord_list = []
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
        color_map = ["red"] * len(elements)
    else:
        color_map = []
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
    fig, ax = plt.subplots(1)
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


def active_subset_random(grid, bias, depth=3) -> list:
    """
    Creates a dictionary of random active assignments.
    :param depth: int
        the number of active elements to return
    :param grid: dict
        hex element dictionary
    :param bias: float
        a value between 0.0 and 1.0 indicating the fraction of grid elements
        that should be active
    :return: dict
        a dictionary containing the element id and a boolean indicator of
        activity
    """
    return np.random.choice(len(grid), size=depth, replace=False).tolist()


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

    graph = nx.Graph()
    for edge in pair_gen:
        if test_element_adjacency(elements, edge):
            graph.add_edge(*edge)

    return graph


def plot_network(network, active=None) -> object:
    """
    Plot the network with the subset highlighted.
    :param active: dict
        dictionary of active elements, must be same length as elements
    :param network: object
        NetworkX network object
    :return: object
        The network with updated colours
    """
    # handle colours
    if not active:
        color_map = ["green"] * len(network)
    else:
        color_map = []
        for node in network:
            if active[node]:
                color_map.append('blue')
            else:
                color_map.append('red')

    pos = nx.spring_layout(network)
    nx.draw(network, pos=pos, node_color=color_map, with_labels=True)
    plt.show()
    return network


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
    # if not active:
    #     subset_nodes = range(len(network))
    # else:
    #     subset_nodes = []
    #     for node in network:
    #         if active[node]:
    #             subset_nodes.append(node)

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
        starting_node = nx.center(net)[0]

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


# TODO compare spatial patterns of isomorphs
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
        flag = 0
        active_subset = active_subset_markov_random(network, depth=depth)
        new_subnet = get_active_sub_network(network, active=active_subset)
        for idx, isomprph in isomprphs.items():
            target_subnet = get_active_sub_network(network, active=isomprph[0])
            if test_network_isomorphs(new_subnet, target_subnet):
                isomprphs[idx].append(active_subset)
                flag = 1
        if flag == 0:
            isomprphs[counter] = [active_subset]
            counter += 1

    return isomprphs


# ToDo find all unique active subsets Create Generator.


if __name__ == '__main__':
    print("Start Mesher.py")
    # np.random.seed(2020)

    # Generate a new hex grid
    myMesh = make_hex_grid(2)
    # plot_hex(myMesh)

    # generate an active subset
    # active1 = active_subset_random(myMesh, 3)
    # active2 = active_subset_random(myMesh, 3)


    # plot hex grid
    # plot_hex(myMesh, active=active1)

    # create a graph and plot it
    net = make_hex_network(myMesh)
    # plot_network(net, active=active1)

    # create a sub-network with only the active elements
    # subnet1 = get_active_sub_network(net, active=active1)
    # subnet2 = get_active_sub_network(net, active=active2)
    # plot_network(subnet1)

    # test that the active set are all connected
    # print("Is subnet 1 fully connected?", test_network_connectivity(subnet1))
    # print("Is subnet 2 fully connected?", test_network_connectivity(subnet2))

    # print(nx.is_isomorphic(subnet1, subnet1))
    # print("Are subnet 1 and subnet 2 isomorphs?",
    #       test_network_isomorphs(subnet1, subnet2))

    active3 = active_subset_markov_random(net, starting_node=None, depth=2)
    # plot_hex(myMesh, active=active3)

    isos = find_all_isomorphs(net, depth=4, runs=10)
    print(isos)

    plot_hex(myMesh)

    # subnet1 = get_active_sub_network(net, active=[9,5])
    # subnet2 = get_active_sub_network(net, active=[9,10])
    # print("Are subnet 1 and subnet 2 isomorphs?",
    #       test_network_isomorphs(subnet1, subnet2))
    # plot_network(subnet1)
    # plot_network(subnet2)

    print("End Mesher.py")
