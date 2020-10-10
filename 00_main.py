# -*- coding: utf-8 -*-
"""
Reinforcement Learning based Process Optimization and Strategy Development in
Conventional Tunneling; G.H. Erharter, T.F. Hansen, Z. Liu, T. Marcher
more publication info...

main code that runs the agent in the environment / brings everything together

Created on Tue Jun 16 22:25:02 2020
code contributors: G.H. Erharter
"""

import numpy as np

import A_utilities
import B_generator
import C_geotechnician
import D_tunnel
import E_plotter

###############################################################################
# fixed variables, hyperparameters & constants (capital letters)

# rockmass hyperparams
TUNNEL_LEN = 200  # total length of tunnel until breakthrough  [m]
RESOLUTION = 10  # factor that multiplies TUNNEL_LEN; 10 => dm
MAX_DIST = 50  # max. allowed distance between TH and bench  [m]
N_CLASSES = 2  # number of rockmass classes

# identify the action - codes
TH_1_0, TH_1_2, TH_5_0, TH_5_2 = 110, 112, 150, 152  # top heading
B_0_0, B_0_2, B_2_0, B_2_2 = 200, 202, 220, 222  # bench & invert

# episode parameters
EPISODES = 120_001  # total number of episodes to go through
MAX_EP_LENGTH = 200  # max. allowed number of steps per episode
SHOW_EVERY = 1_000  # make a plot, rendering and save every n episode
PRINT_EVERY = 100  # print progress every n episode
PREV_EP = 0  # number of starting episode (0 for a fresh start)
STATS_SAVEPATH = r'02_plots\episode_stats.csv'  # path to save the stats file

# agent's hyperparameters
epsilon = 1  # initial exploration
MIN_EPSILON = 0.05  # final exploration -> reached after 99858 eps with decay = 0.99997
DISCOUNT = 0.99
EPSILON_DECAY = 0.99997  # Every episode will be epsilon*EPS_DECAY

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

if PREV_EP == 0:
    checkpoint = None
else:
    checkpoint = fr'04_checkpoints\ESA_ep{PREV_EP}.h5'

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
                                 DISCOUNT=DISCOUNT, checkpoint=checkpoint)
pltr = E_plotter.plotter()

###############################################################################

# create / get dataframe that tracks stats of each episode
df = utils.master_stats_dataframe(STATS_SAVEPATH, start_episode=PREV_EP)

# main loop that iterates over all episodes
for episode in range(PREV_EP, PREV_EP+EPISODES):
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
        if episode % SHOW_EVERY == 0 and episode > PREV_EP:
            # renders individual frames to finally render whole episode
            pltr.render_frame(tunnel.geo_section, tunnel.sup_section,
                              fr'02_plots\tmp\{step-1}.png')

        # epsilon greedy action selection policy
        if np.random.random() > epsilon:  # query a model for Q values
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

        # Every step we update replay memory and train main network
        agent.update_replay_memory((current_state, action, reward, new_state,
                                    done))
        hist = agent.train(done, step)
        try:
            losses_.append(hist.history['loss'][0])
            accuracies_.append(hist.history['acc'][0])
        except AttributeError:
            pass

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
    df_ep = utils.ep_stats_dataframe(episode, rewards, epsilon, ep_pf, losses_,
                                     accuracies_, instabilites, frac_rt1_rt2,
                                     tunnel.pos_th, tunnel.pos_bi,
                                     max(dists_th_bi), step, term, actions)
    df = df.append(df_ep, ignore_index=True)

    # decay epsilon
    epsilon = agent.decay_epsilon(epsilon, MIN_EPSILON, EPSILON_DECAY)

    # get stats and plot progress
    if episode % PRINT_EVERY == 0 and episode > PREV_EP:
        utils.print_status(df, PRINT_EVERY)
    if episode % SHOW_EVERY == 0 and episode > PREV_EP:

        pltr.progress_plot(np.array(pos_ths), np.array(pos_bis),
                           np.array(actions), rewards, 110, 112, 150, 152,
                           200, 202, 220, 222, episode,
                           fr'02_plots\episode_{episode}_sample.png')

        pltr.reward_plot(df,
                         savepath=fr'02_plots\episode_{episode}_rewards.png',
                         windows=100, plot_eprewpoints=True)

        pltr.render_episode(r'02_plots\tmp', fps=2, x_pix=1680, y_pix=480,
                            savepath=fr'02_plots\ep{episode}.avi')

        df.to_csv(STATS_SAVEPATH, index=False)

        agent.save(fr'04_checkpoints\ESA_ep{episode}.h5')
