# -*- coding: utf-8 -*-
"""
Reinforcement Learning based Process Optimization and Strategy Development in
Conventional Tunneling; G.H. Erharter, T.F. Hansen, Z. Liu, T. Marcher
more publication info...

Code that runs one individual checkpoint of an agent without learning to test
its performance. Basic functionality is the same as "00_main.py" and the final
product is to save a file with statistics of the tested episodes for further
analysis / plotting.

Created on Mon Jun  1 08:47:34 2020
code contributors: Georg H. Erharter, Tom F. Hansen
"""

# part of code has to be put on top to choose a specific hardware component for
# running an agent. -> cannot run two agents on same GPU
import os
# The GPU id to use, usually either "0" or "1"; to use CPU "-1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import pandas as pd
from pathlib import Path

import A_utilities
import B_generator
import C_geotechnician
import D_tunnel
import E_plotter


###############################################################################
# fixed variables, hyperparameters, constants (capital letters)

# rockmass hyperparams
TUNNEL_LEN = 200  # total length of tunnel until breakthrough  [m]
RESOLUTION = 10  # factor that multiplies TUNNEL_LEN; 10 => dm
MAX_DIST = 50  # max. allowed distance between TH and bench  [m]
N_CLASSES = 2  # number of rockmass classes

# identify the action - codes
TH_1_0, TH_1_2, TH_5_0, TH_5_2 = 110, 112, 150, 152  # top heading
B_0_0, B_0_2, B_2_0, B_2_2 = 200, 202, 220, 222  # bench & invert

# episode parameters
EPISODES = 100  # total number of episodes to go through
MAX_EP_LENGTH = 200  # max. allowed number of steps per episode
PRINT_EVERY = 10  # print progress every n episode

# agent's hyperparameters
# we found that better results during testing can be achieved by setting
# EPSILON > 0. See paper chapter....
EPSILON = 0.05
DISCOUNT = 0.99

# path to saved model checkpoint
CHECKPOINT_PATH = Path('04_checkpoints/2020_10_09/ESA_ep21000.h5')
# path to where the statistics of the test run should be saved
TESTSTATS_PATH = Path('06_results/stats.csv')

###############################################################################
# dictionaries

# contains all rewards and penalties
rew_dict = {'breakthrough': TUNNEL_LEN * 3, 'timeout': -TUNNEL_LEN * 3,
            'wrong_sequence': -6, 'instability': -5, 'change_method': -4,
            'distance_th_bench': -3, 'support_usage': -2, 'move': -1}

# assignment of action codes to cutting & face support lengths
cutting_lengths = {TH_1_0: 2, TH_1_2: 2, TH_5_0: 4, TH_5_2: 4,
                   B_0_0: 2, B_0_2: 2, B_2_0: 4, B_2_2: 4}
support_lengths = {TH_1_0: 0, TH_1_2: 10, TH_5_0: 0, TH_5_2: 10,
                   B_0_0: 0, B_0_2: 10, B_2_0: 0, B_2_2: 10}

# measures of top heading & bench excavation
exc_dict = {'TopHead. area [m²]': 55.56, 'Bench area [m²]': 32.57,
            'TopHead. equiv. D [m]': 8.63, 'Bench equiv. D [m]': 6.46}

# bad rockmass type
rockmass_dict1 = {'spec. weight [N/m³]': 24000, 'cohesion [Pa]': 23000,
                  'friction angle [°]': 20}

# good rockmass type
rockmass_dict2 = {'spec. weight [N/m³]': 25000, 'cohesion [Pa]': 40000,
                  'friction angle [°]': 30}

###############################################################################
# computed variables based on fixed variables and dictionaries

# total length of observation space as number of datapoints
n_datapoints = (TUNNEL_LEN + max(support_lengths.values())) * RESOLUTION
observation_space_values = (2, n_datapoints, 2)  # shape of observation space
# maximum length, beyond which no more updates of the position of top head. & bench are donce
max_pos = (TUNNEL_LEN + max(cutting_lengths.values())) * RESOLUTION

rockmass_dicts = [rockmass_dict1, rockmass_dict2]

###############################################################################
# instantiations

utils = A_utilities.utilities(N_CLASSES)
gen = B_generator.generator(n_datapoints)
gt = C_geotechnician.geotechnician()
agent = C_geotechnician.DQNAgent(observation_space_values,
                                 list(cutting_lengths.keys()),
                                 DISCOUNT=DISCOUNT, checkpoint=CHECKPOINT_PATH)
pltr = E_plotter.plotter()

###############################################################################

# create dataframe that tracks stats of each testing episode
try:  # remove prev. stats df if existing
    os.remove(TESTSTATS_PATH)
    df = utils.master_stats_dataframe(TESTSTATS_PATH)
except FileNotFoundError:
    df = utils.master_stats_dataframe(TESTSTATS_PATH)

# main loop that iterates over all episodes
for episode in range(EPISODES):
    a = TH_1_2  # initial action is top heading supported with 10m support
    step = 1  # counter of steps in every episode
    instabilites = 0  # counter for how many faces are instable
    ep_pf = 0  # initial face pressure of episode
    done = False  # one episode continues until done = True

    # empty lists that collect statistics of each episode
    actions = []  # all actions used during episode
    pos_ths = []  # all positions of the top heading excavation
    pos_bis = []  # all positions of bench excavation
    dists_th_bi = []  # distances between top heading and bench
    rewards = []  # development of reward over the episode
    losses_ = []  # ANN loss after each prediction
    accuracies_ = []  # ANN acuracy after each prediction

    # generate new / unique geology for episode
    rockmass_types = gen.generate_rock_types(N_CLASSES) + 1
    # instantiate new tunnel sections
    tunnel = D_tunnel.tunnel(TUNNEL_LEN + max(support_lengths.values()),
                             RESOLUTION, cutting_lengths, support_lengths)
    tunnel.update_sections(rockmass_types, a)
    # get geological section and support section at beginning
    geo_section, sup_section = tunnel.geo_section, tunnel.sup_section
    # build the ANN's input
    current_state = utils.ANN_input(geo_section, sup_section)

    # excavate until a terminal state is reached (= breakthrough or timeout)
    while not done:
        # eventually render episode
        # if episode == EPISODES-1:
        #     # renders individual frames to finally render whole episode
        #     pltr.render_frame(tunnel.geo_section, tunnel.sup_section,
        #                       fr'06_results\tmp\{step-1}.png')

        # epsilon greedy action selection policy
        if np.random.random() > EPSILON:  # query a model for Q values
            action = np.argmax(agent.get_qs(current_state))
        else:  # Get random action
            action = np.random.randint(0, len(cutting_lengths.keys()))
        a = list(cutting_lengths.keys())[action]  # get eaction code

        # update positions of excavation unless there is already a breakthrough
        tunnel.update_positions(a, max_pos)

        # check stability dependent on action;
        if a < 200:  # top heading
            pf = gt.check_stability(sup_section[0, :], tunnel.pos_th,
                                    exc_dict['TopHead. equiv. D [m]'],
                                    cutting_lengths[a],
                                    rockmass_dicts[rockmass_types[tunnel.pos_th]-1])
        elif a >= 200:  # bench
            pf = gt.check_stability(sup_section[1, :], tunnel.pos_bi,
                                    exc_dict['Bench equiv. D [m]'],
                                    cutting_lengths[a],
                                    rockmass_dicts[rockmass_types[tunnel.pos_bi]-1])

        # handle rewards
        reward = tunnel.handle_rewards(TUNNEL_LEN, rew_dict, step,
                                       MAX_EP_LENGTH, pf, actions, a, MAX_DIST)

        # episode abortion criterion
        if reward == rew_dict['breakthrough']:
            done, term = True, 'breakthrough'
        elif reward == rew_dict['timeout']:
            done, term = True, 'timeout'

        # update state based on taken action
        tunnel.update_sections(rockmass_types, a)
        geo_section, sup_section = tunnel.geo_section, tunnel.sup_section
        new_state = utils.ANN_input(geo_section, sup_section)

        current_state = new_state
        step += 1
        ep_pf += pf
        if pf >= 0:
            instabilites += 1

        rewards.append(reward)
        actions.append(a)
        pos_ths.append(tunnel.pos_th)
        pos_bis.append(tunnel.pos_bi)
        dists_th_bi.append(tunnel.dist_th_bi)

    # create dataframe with stats of episode and then add to main stats df
    frac_rt1_rt2 = np.unique(rockmass_types, return_counts=True)[1]
    frac_rt1_rt2 = frac_rt1_rt2[0] / frac_rt1_rt2[1]
    df_ep = utils.ep_stats_dataframe(episode, rewards, EPSILON, ep_pf, losses_,
                                     accuracies_, instabilites, frac_rt1_rt2,
                                     tunnel.pos_th, tunnel.pos_bi,
                                     max(dists_th_bi), step, term, actions)
    df = df.append(df_ep, ignore_index=True)

    # get stats and plot progress
    if episode % PRINT_EVERY == 0:
        utils.print_status(df, PRINT_EVERY)
    if episode == EPISODES-1:

        pltr.progress_plot(np.array(pos_ths), np.array(pos_bis),
                           np.array(actions), rewards, 110, 112, 150, 152,
                           200, 202, 220, 222, episode,
                           Path('06_results/sample.png'))

        # pltr.render_episode(r'02_plots\tmp', fps=2, x_pix=1680, y_pix=480,
        #                     savepath=fr'06_results\sample.avi')

        df.to_csv(TESTSTATS_PATH, index=False)
