# -*- coding: utf-8 -*-
"""
Reinforcement Learning based Process Optimization and Strategy Development in
Conventional Tunneling; G.H. Erharter, T.F. Hansen, Z. Liu, T. Marcher
more publication info...

Code that does analysis or plots of test runs, checkpoints etc.

Created on Fri Oct  9 14:49:25 2020
code contributors: Georg H. Erharter, Tom F. Hansen
"""

import pandas as pd
from pathlib import Path
from typing import List

import E_plotter


pltr = E_plotter.plotter()


WINDOW = 500  # size of sliding window for plots
# list of training runs in the "02_plots" folder that should be analyzed
TRAINING_RUNS = ['2020_09_29', '2020_10_04', '2020_10_09', '2020_10_10',
                 '2020_10_17']


##############################################################################
# analysis of training stats of multiple training passes

# first get the parameters that should be plotted
max_eps: List[int] = []
rewards: List[pd.DataFrame] = []
instabilities: List[pd.DataFrame] = []
n_blasts: List[pd.DataFrame] = []
losses: List[pd.DataFrame] = []

for run in TRAINING_RUNS:
    # load data
    df = pd.read_csv(Path(f'02_plots/{run}/episode_stats.csv'))

    # identify where the agent has had its best performance
    idx_max_rew = df['ep. rewards'].rolling(window=WINDOW, center=True).mean().argmax()
    max_rew = round(df['ep. rewards'].rolling(window=WINDOW, center=True).mean().iloc[idx_max_rew], 0)
    print(f'run {run} reached max reward of {max_rew} at episode {idx_max_rew}')

    # identify where number of isntabilities was at its minimum
    idx_min_inst = df['instabilities'].rolling(window=WINDOW, center=True).mean().argmin()
    min_inst = round(df['instabilities'].rolling(window=WINDOW, center=True).mean().iloc[idx_min_inst], 0)
    print(f'run {run} reached min instabilities of {min_inst} at episode {idx_min_inst}')

    # identify where number of blasts per episode was at its minimum
    idx_min_blast = df['blasts per breakthrough'].rolling(window=WINDOW, center=True).mean().argmin()
    min_blasts = round(df['blasts per breakthrough'].rolling(window=WINDOW, center=True).mean().iloc[idx_min_inst], 0)
    print(f'run {run} reached min blasts of {min_blasts} at episode {idx_min_blast}\n')

    max_eps.append(df['episode'].max()+1)
    rewards.append(df['ep. rewards'])
    instabilities.append(df['instabilities'])
    n_blasts.append(df['blasts per breakthrough'])
    losses.append(df['ep. loss'])

# plot the parameters
pltr.multi_agent_plot(max_eps, rewards, instabilities, n_blasts, losses,
                      savepath=Path('06_results/model_comparison.jpg'),
                      window=WINDOW)


##############################################################################
# analysis of test statistics of one agent

df = pd.read_csv(Path('06_results/sample_2020_10_10_ep119000.csv'))

print('\nstatistics: min. max. median')
print('reward:', df['ep. rewards'].min(), df['ep. rewards'].max(),
      df['ep. rewards'].median())
print('instabilities:', df['instabilities'].min(), df['instabilities'].max(),
      df['instabilities'].median())
print('max. dist. th-b [m]:', df['max. dist th-bi'].min()/10,
      df['max. dist th-bi'].max()/10, df['max. dist th-bi'].median()/10)
print('blasts per ep.:', df['blasts per breakthrough'].min(),
      df['blasts per breakthrough'].max(),
      df['blasts per breakthrough'].median())

# histograms for statistics of a tested agent
pltr.test_stats_histograms(df, Path('06_results/histograms.jpg'))

# boxplot of the actions that the agent took
pltr.test_stats_boxplot(df, Path('06_results/boxplot.jpg'))
