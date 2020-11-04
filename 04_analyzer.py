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


# ##############################################################################
# # analysis of training stats of one training pass

# df = pd.read_csv(Path('02_plots/episode_stats.csv'))

# idx_max_rew = df['ep. rewards'].argmax()
# max_rew = int(df['ep. rewards'].iloc[idx_max_rew])
# print(f'agent reached max reward of {max_rew} at episode {idx_max_rew}')


##############################################################################
# analysis of test statistics of one agent

df = pd.read_csv(Path('06_results/stats.csv'))

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


##############################################################################
# analysis of training stats of multiple training passes

# list of training runs that should be visualized. Respective training
# statistics files  should be saved in the folder 02_plots.
training_runs = ['2020_09_29', '2020_10_04', '2020_10_09', '2020_10_10',
                 '2020_10_17']

# first get the parameters that should be plotted
max_eps: List[int] = []
rewards: List[pd.DataFrame] = []
instabilities: List[pd.DataFrame] = []
n_blasts: List[pd.DataFrame] = []
losses: List[pd.DataFrame] = []

for run in training_runs:
    df = pd.read_csv(Path(f'02_plots/{run}/episode_stats.csv'))
    max_eps.append(df['episode'].max()+1)
    rewards.append(df['ep. rewards'])
    instabilities.append(df['instabilities'])
    n_blasts.append(df['blasts per breakthrough'])
    losses.append(df['ep. loss'])

# plot the parameters
pltr.multi_agent_plot(max_eps, rewards, instabilities, n_blasts, losses,
                      savepath=Path('06_results/model_comparison.jpg'),
                      window=500)
