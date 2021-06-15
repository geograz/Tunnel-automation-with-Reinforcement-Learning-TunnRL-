# -*- coding: utf-8 -*-
"""
Reinforcement Learning based Process Optimization and Strategy Development in
Conventional Tunneling; G.H. Erharter, T.F. Hansen, Z. Liu, T. Marcher
DOI: https://doi.org/10.1016/j.autcon.2021.103701

Code that generates a new and unique geological section at the beginning of
every episode. = part of the environment.

Created on Wed Jul  1 15:29:52 2020
code contributors: Georg H. Erharter, Tom F. Hansen
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


class generator():

    def __init__(self, n_datapoints: int):
        self.n_dp = n_datapoints  # number of datapoints

    def gen_rand_walk(self) -> np.array:
        """
        function generates a random walk with boundaries
        https://www.geeksforgeeks.org/random-walk-implementation-python/
        """
        prob = [0.05, 0.95]  # Probability to move up or down

        # statically defining the starting position
        start = 50
        positions = [start]

        # creating the random points
        rr = np.random.random(self.n_dp)
        downp = rr < prob[0]
        upp = rr > prob[1]

        for idownp, iupp in zip(downp, upp):
            down = idownp and positions[-1] > 1
            up = iupp and positions[-1] < 100
            positions.append(positions[-1] - down + up)

        return np.array(positions[:-1])

    def normalize(self, data: np.array) -> np.array:
        """min - max scaling of numpy array"""
        data = data - data.min()  # shift min to 0
        data = data / data.max()  # set max to 1
        return data

    def generate_rock_types(self, N_CLASSES: int, return_rand_walk=False) -> np.array:
        """ function transforms the random walk into categorical rockmass
        types """
        rand_walk = self.gen_rand_walk()
        rand_walk_norm = self.normalize(rand_walk)
        rock_types = rand_walk_norm*(N_CLASSES-1)
        if return_rand_walk is True:
            return np.round(rock_types, 0).astype(int), rand_walk_norm
        else:
            return np.round(rock_types, 0).astype(int)


if __name__ == '__main__':

    # exemplary plot of the random walk -> for paper chapter 3.1
    import D_tunnel
    import E_plotter

    np.random.seed(7)  # fix seed for reproducibility

    ###########################################################################
    # identify the actions and get other fixed variables and dictionaries
    TH_1_0, TH_1_2 = 110, 112
    TH_5_0, TH_5_2 = 150, 152

    B_0_0, B_0_2 = 200, 202
    B_2_0, B_2_2 = 220, 222

    cutting_lengths = {TH_1_0: 2, TH_1_2: 2, TH_5_0: 4, TH_5_2: 4,
                       B_0_0: 2, B_0_2: 2, B_2_0: 4, B_2_2: 4}
    support_lengths = {TH_1_0: 0, TH_1_2: 10, TH_5_0: 0, TH_5_2: 10,
                       B_0_0: 0, B_0_2: 10, B_2_0: 0, B_2_2: 10}

    TUNNEL_LEN, RESOLUTION, ADDITIONAL, N_CLASSES = 200, 10, 10, 2

    ###########################################################################
    # generate the data

    # instantiate generator and the plotter
    gen = generator(TUNNEL_LEN * RESOLUTION + ADDITIONAL * RESOLUTION)
    pltr = E_plotter.plotter()

    # generate the geological section
    rock_types, data_norm = gen.generate_rock_types(2, return_rand_walk=True)

    tunnel = D_tunnel.tunnel(TUNNEL_LEN + ADDITIONAL, RESOLUTION,
                             cutting_lengths, support_lengths)
    tunnel.pos_th = 1650  # fixed position of top heading excavation
    tunnel.pos_bi = 1250  # fixed position of bench excavation
    tunnel.update_sections(rock_types + 1, TH_1_0)
    geo_section = tunnel.geo_section

    # plot the figure
    fig = plt.figure(figsize=(14, 4))

    ax = fig.add_subplot(2, 1, 1)
    pltr.render_geo_section(geo_section)
    ax.axvline(2000, color='white')
    ax.text(x=2010, y=1.1, s='breakthrough', color='white', rotation=90,
            fontsize=12)

    ax = fig.add_subplot(2, 1, 2)
    ax.plot(np.arange(len(data_norm)), data_norm, color='black')
    ax.axhline(0.5, color='black', ls='--')

    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels([0, 0.5, 1])
    ax.set_xlim(left=0, right=len(data_norm))
    ax.set_xlabel('tunnellength [dm]', fontsize=12)
    ax.set_ylabel('random walk', fontsize=12)

    plt.tight_layout()
    plt.savefig(Path('02_plots/00_data_generator.jpg'), dpi=600)
