# -*- coding: utf-8 -*-
"""
Reinforcement Learning based Process Optimization and Strategy Development in
Conventional Tunneling; G.H. Erharter, T.F. Hansen, Z. Liu, T. Marcher
DOI: https://doi.org/10.1016/j.autcon.2021.103701

Part of the environment that does the updating of positions of the excavation,
"reveals" new geology and also handles the reward system.

Created on Mon Jul 20 10:47:08 2020
code contributors: Georg H. Erharter, Tom F. Hansen
"""

import numpy as np
from typing import List


class tunnel():

    def __init__(self, TUNNEL_LEN: int, RESOLUTION: int,
                 cutting_lengths: dict, support_lengths: dict):
        self.cutting_lengths = cutting_lengths
        self.support_lengths = support_lengths
        self.RES = RESOLUTION
        self.TUNNEL_LEN = TUNNEL_LEN * RESOLUTION  # != breakthrough length

        self.pos_th, self.pos_bi = 0, 0  # init. position of top head. & bench

        # geological section
        self.geo_section = np.full((2, self.TUNNEL_LEN), 0)
        # section with face supports
        self.sup_section = np.full((2, self.TUNNEL_LEN), 0)

    def update_positions(self, action: int, max_pos: int) -> None:
        ''' function updates the position of the front of the excavation (i.e.
        tunnel face) based on the chosen action of the agent '''
        if self.pos_th > max_pos and action < 200:
            pass
        elif self.pos_bi > max_pos and action >= 200:
            pass
        # only update if position is not beyond breakthrough
        else:
            if action < 200:  # top heading
                self.pos_th += self.cutting_lengths[action]*self.RES
            elif action >= 200:  # bench
                self.pos_bi += self.cutting_lengths[action]*self.RES
            self.dist_th_bi = np.abs(self.pos_th - self.pos_bi)

    def update_sections(self, rockmass_types: np.array, action: int) -> None:
        ''' function updates the geological section and the support section
        which serves as the RL agent's state '''
        # update top heading
        self.geo_section[0, :][:self.pos_th] = rockmass_types[:self.pos_th]
        # update bench
        self.geo_section[1, :][:self.pos_bi] = rockmass_types[:self.pos_bi]

        # update support section
        if action < 200:  # top heading
            self.sup_section[0, :][self.pos_th:self.pos_th+self.support_lengths[action]*self.RES] = 1

        elif action >= 200:  # bench
            self.sup_section[1, :][self.pos_bi:self.pos_bi+self.support_lengths[action]*self.RES] = 1

    def handle_rewards(self, break_len: int, rew_dict: dict, step: float,
                       MAX_EP_LENGTH: int, pf: float, actions: List, a: float,
                       MAX_DIST: int) -> int:
        ''' function handles the rewards / gives points either based on the
        last action or the current state; penalties are ordered depending on
        their severity '''
        # check if breakthrough was achieved
        if self.pos_th > break_len * self.RES and self.pos_bi > break_len * self.RES:
            reward = rew_dict['breakthrough']
        # check if a timeout has occured
        elif step >= MAX_EP_LENGTH:
            reward = rew_dict['timeout']
        # check for bench in front of top heading -> wrong exc. sequence
        elif self.pos_bi > self.pos_th:
            reward = rew_dict['wrong_sequence']
        # check for face stability
        elif pf >= 0:
            reward = rew_dict['instability']
        # penalize change from topheading to bench
        elif step > 1 and actions[-1] < 200 and a >= 200:
            reward = rew_dict['change_method']
        # penalize change from bench to topheading
        elif step > 1 and actions[-1] >= 200 and a < 200:
            reward = rew_dict['change_method']
        # check if distance top heading - bench is too big
        elif np.abs(self.pos_th - self.pos_bi) > MAX_DIST * self.RES:
            reward = rew_dict['distance_th_bench']
        # check if face support was used
        elif self.support_lengths[a] > 0:
            reward = rew_dict['support_usage']
        # small penalty for every other step to encourage fast work
        else:
            reward = rew_dict['move']

        return reward
