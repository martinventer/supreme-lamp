# Mesher

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import RegularPolygon

import networkx as nx
import itertools

def make_hex_grid(dimension) -> dict:
    """
    Create a dictionary of hexagonal grid elements.
    :param dimension: int
        radial dimension for the regular hex grid
    :return: dict
        {element_ID: [x-coord, y-coord, z-coord], }
    """
    coord_list = []
    for x in range(-1*dimension, dimension +1, 1):
        for y in range(-1*dimension, dimension +1, 1):
            for z in range(-1*dimension, dimension +1, 1):
                if sum([x, y, z]) is 0:
                    coord_list.append([x, y, z])

    return {id: coord for id, coord in enumerate(coord_list)}


def hex_to_cart(hex_coords) -> (list, list):
    """
    Takes a list of hex coords and converts them to a list of cartesian
    coordinates.
    :param hex_coords: list
        list of hex coordinates [[x, y, z], ]
    :return: (list, list)
        pair of coordinate lists x and y ([x], [y])
    """
    x_coords = [c[0] for c in hex_coords]
    y_coords = [2. * np.sin(np.radians(60)) * (c[1] - c[2]) / 3. for c in
                hex_coords]

    return x_coords, y_coords


def plot_hex(elements, cols=None) -> None:
    """
    Plots a hex grid with color and label overlay. Conversion of hexagon to
    cartesian coordinates.
    :param elements: dict
        dictionary of hex elements
    :param cols: dict
        dictionary of active elements, must be same length as elements
    :return: None
    """
    # handle missing colours
    if not cols:
        cols = ["blue"] * len(elements)
    else:
        for elem, active in cols.items():
            if active:
                cols[elem] = "blue"
            else:
                cols[elem] = "red"
        cols = cols.values()

    # create lable and coord  list
    lables = elements.keys()
    hex_coords = elements.values()

    # convert from hex [x, y, z] to cart [x, y]
    x_coords, y_coords = hex_to_cart(hex_coords)

    # process the figure
    fig, ax = plt.subplots(1)
    ax.set_aspect('equal')

    for x, y, c, l in zip(x_coords, y_coords, cols, lables):
        colour = c[0].lower() # makes sure that colour is are lower ca
        hex = RegularPolygon((x, y),
                             numVertices=6,
                             radius=2. / 3.,
                             orientation=np.radians(30),
                             facecolor=colour,
                             alpha=0.2,
                             edgecolor='k')

        ax.add_patch(hex)
        # Also add a text label
        ax.text(x, y + 0.2, l, ha='center', va='center', size=20)

    # Also add scatter points in hexagon centres
    ax.scatter(x_coords, y_coords, c=[c[0].lower() for c in cols], alpha=0.5)

    plt.show()

    return None


def active_subset_random(grid, bias) -> dict:
    """
    Creates a dictionary of random active assignments.
    :param grid: dict
        hex elemenet dictionary
    :param bias: float
        a value between 0.0 and 1.0 indicating the fraction of grid elements
        that should be active
    :return: dict
        a dictionary containing the element id and a boolean indicator of
        activity
    """
    return {
        ids: active for ids, active in enumerate(
            np.random.choice([False, True], len(grid), p=[1-bias, bias])
        )
    }


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
    element_A = elements[pair[0]]
    element_B = elements[pair[1]]
    difference = [a - b for a, b in zip(element_A, element_B)]
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
        retuns a network object
    """
    pair_gen = unique_element_pairs(elements)

    G = nx.Graph()
    for edge in pair_gen:
        if test_element_adjacency(elements, edge):
            G.add_edge(*edge)

    return G


def plot_network(net) -> None:
    nx.draw(net, with_labels=True)
    plt.show()
    return None

# ToDo test the active subset for adjacency.
# ToDo MCMC to find all active subsets Create Generator.
# ToDo Find acitivity subsets that are adjacent


if __name__ == '__main__':
    print("Start Mesher.py")

    myMesh = make_hex_grid(2)

    active = active_subset_random(myMesh, 0.1)

    plot_hex(
        myMesh,
        cols=active
    )

    net = make_hex_network(myMesh)
    plot_network(net)

    print("End Mesher.py")