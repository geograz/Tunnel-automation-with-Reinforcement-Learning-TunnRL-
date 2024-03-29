# -*- coding: utf-8 -*-
"""
Reinforcement Learning based Process Optimization and Strategy Development in
Conventional Tunneling; G.H. Erharter, T.F. Hansen, Z. Liu, T. Marcher
DOI: https://doi.org/10.1016/j.autcon.2021.103701

class of mixed utilities that don't fit in any other classes

Created on Wed Jul  1 15:29:00 2020
code contributors: Georg H. Erharter, Tom F. Hansen
"""

import numpy as np
import pandas as pd
from typing import List


class utilities():

    def __init__(self, N_CLASSES: int):
        self.N_CLASSES = N_CLASSES  # number of rockmass classes

    def master_stats_dataframe(self, savepath: str,
                               start_episode: pd.DataFrame=None) -> pd.DataFrame:
        """
        Function that either gets an existing dataframe with already
        recorded edepisode statistics or creates a new one.
        """
        try:
            df = pd.read_csv(savepath)
            # drop last row to avoid double rows
            df.drop(df[df['episode'] >= start_episode].index, inplace=True)
        except FileNotFoundError:
            df = pd.DataFrame({'episode': [], 'ep. rewards': [],
                               'epsilons': [], 'ep. pf': [], 'ep. loss': [],
                               'ep. accuracy': [], 'instabilities': [],
                               'frac. RT1:RT2': [],
                               'reached pos. top.head.': [],
                               'reached pos. bench': [],
                               'max. dist th-bi': [],
                               'blasts per breakthrough': [], 'terminals': [],
                               '110': [], '112': [], '150': [], '152': [],
                               '200': [], '202': [], '220': [], '222': []})
        return df

    def ep_stats_dataframe(self, episode: float, rewards: List, epsilon: float,
                           ep_pf: float, losses_: List, accuracies_: List,
                           instabilites: float, frac_rt1_rt2: float,
                           pos_th: float, pos_bi: float, dist_th_bi: float,
                           step: float, term: str,
                           actions: List) -> pd.DataFrame:
        """create and return dataframe with statistics of one episode that will
        then be appended to the main episode statistics dataframe.
        """
        df = pd.DataFrame({'episode': [episode], 'ep. rewards': [sum(rewards)],
                           'epsilons': [epsilon], 'ep. pf': ep_pf,
                           'ep. loss': [np.mean(losses_)],
                           'ep. accuracy': [np.mean(accuracies_)],
                           'instabilities': [instabilites],
                           'frac. RT1:RT2': [frac_rt1_rt2],
                           'reached pos. top.head.': [pos_th],
                           'reached pos. bench': [pos_bi],
                           'max. dist th-bi': [dist_th_bi],
                           'blasts per breakthrough': [step],
                           'terminals': [term],
                           '110': [0], '112': [0], '150': [0], '152': [0],
                           '200': [0], '202': [0], '220': [0], '222': [0]})

        # count how many times each action was used during episode
        ep_actions, ep_counts = np.unique(actions, return_counts=True)
        ep_actions = np.char.mod('%d', ep_actions)
        for i, ep_action in enumerate(ep_actions):
            df[ep_action] = ep_counts[i]

        return df

    def ANN_input(self, geo_section: np.array,
                  sup_section: np.array) -> np.array:
        """function creates the hypermatrix / array that is the agent's
        input"""
        section = geo_section / self.N_CLASSES
        return np.dstack((section, sup_section))

    def print_status(self, df: pd.DataFrame, PRINT_EVERY: int) -> None:
        """function prints the status of the training progress of the current
        episode to the console"""
        episode = int(df['episode'].iloc[-1])
        epsilon = round(df['epsilons'].iloc[-1], 4)
        pr_rew = int(np.round(df['ep. rewards'].iloc[-PRINT_EVERY:].mean(), 0))
        pr_loss = np.round(df['ep. loss'].iloc[-PRINT_EVERY:].mean(), 3)
        pr_acc = np.round(df['ep. accuracy'].iloc[-PRINT_EVERY:].mean(), 3)
        pr_pos_th = int(np.round(df['reached pos. top.head.'].iloc[-PRINT_EVERY:].mean(), 0))
        pr_pos_bi = int(np.round(df['reached pos. bench'].iloc[-PRINT_EVERY:].mean(), 0))
        pr_n_blasts = int(np.round(df['blasts per breakthrough'].iloc[-PRINT_EVERY:].mean(), 0))
        pr_instable = np.round(df['instabilities'].iloc[-PRINT_EVERY:].mean(), 1)
        # pr_ep_pf = int(np.round(df['ep. pf'].iloc[-PRINT_EVERY:].mean(), 0))

        terminals, counts = np.unique(df['terminals'].iloc[-PRINT_EVERY:],
                                      return_counts=True)
        pr_terminal = terminals[counts.argmax()]

        actions = ['110', '112', '150', '152', '200', '202', '220', '222']
        a_most_used = df[actions].iloc[-PRINT_EVERY:].mean(axis=0).idxmax()

        print(f'\nep:{episode}, eps:{epsilon}, rew:{pr_rew}, loss:{pr_loss}, acc:{pr_acc}')
        print(f'pos th/bi: {pr_pos_th}/{pr_pos_bi}, n blasts:{pr_n_blasts}, instable:{pr_instable}, act:{a_most_used}, terminals:{pr_terminal}')
