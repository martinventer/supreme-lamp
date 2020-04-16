# Mesher

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import RegularPolygon


def hex_plot(coords, colours=None, labels=None) -> None:
    """
    Plots a hex grid with color an label overlay. Conversion of hexagon to
    cartesian coordinates.
    :param coords:
    :param colours:
    :param labels:
    :return: None
    """
    # handle missing lables and colurs
    if not colours:
        colours = ["blue"]*len(coords)

    if not labels:
        labels = ["-"]*len(coords)

    # convert from hex [x, y, z] to cart [x, y]
    x_coords = [c[0] for c in coords]
    y_coords = [2. * np.sin(np.radians(60)) * (c[1] - c[2]) / 3. for c in coords]

    fig, ax = plt.subplots(1)
    ax.set_aspect('equal')

    for x, y, c, l in zip(x_coords, y_coords, colours, labels):
        colour = c[0].lower() # makes sure that colour identifires are lower ca
        hex = RegularPolygon((x, y),
                             numVertices=6,
                             radius=2. / 3.,
                             orientation=np.radians(30),
                             facecolor=colour,
                             alpha=0.2,
                             edgecolor='k')

        ax.add_patch(hex)
        # Also add a text label
        ax.text(x, y + 0.2, l[0], ha='center', va='center', size=20)

    # Also add scatter points in hexagon centres
    ax.scatter(x_coords, y_coords, c=[c[0].lower() for c in colours], alpha=0.5)

    plt.show()


def make_grid(x_step, y_step, z_step) -> list:
    coord_list = []
    for x in range(-1*x_step, x_step +1, 1):
        for y in range(-1*y_step, y_step +1, 1):
            for z in range(-1*z_step, z_step +1, 1):
                if sum([x, y, z]) is 0:
                    coord_list.append([x, y, z])

    return coord_list


# ToDo write grid hex grid generator
# ToDo write hex grid plotter

if __name__ == '__main__':
    print("aaaa")

    # coord = [[0, 0, 0], [0, 1, -1], [-1, 1, 0], [-1, 0, 1], [0, -1, 1],
    #          [1, -1, 0], [1, 0, -1]]
    # colors = [["Green"], ["Blue"], ["Green"], ["Green"], ["Red"], ["Green"],
    #           ["Green"]]
    # labels = [['yes'], ['no'], ['yes'], ['no'], ['yes'], ['no'], ['no']]

    hex_plot(make_grid(3, 3, 3))

    # print(make_grid(1,1,1))