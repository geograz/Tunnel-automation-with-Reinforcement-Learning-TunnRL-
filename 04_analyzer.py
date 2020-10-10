# -*- coding: utf-8 -*-
"""
Reinforcement Learning based Process Optimization and Strategy Development in
Conventional Tunneling; G.H. Erharter, T.F. Hansen, Z. Liu, T. Marcher
more publication info...

Code that does analysis or plots of test runs, checkpoints etc.

Created on Fri Oct  9 14:49:25 2020
code contributors: G.H. Erharter
"""

import pandas as pd

import E_plotter


pltr = E_plotter.plotter()

##############################################################################
# analysis of training stats

df = pd.read_csv(r'02_plots\episode_stats.csv')

idx_max_rew = df['ep. rewards'].argmax()
max_rew = int(df['ep. rewards'].iloc[idx_max_rew])
print(f'agent reached max reward of {max_rew} at episode {idx_max_rew}')

##############################################################################
# analysis of test statistics

df = pd.read_csv(r'06_results\stats.csv')

print('\nstatistics: min. max. median')
print('reward:', df['ep. rewards'].min(), df['ep. rewards'].max(),
      df['ep. rewards'].median())
print('instabilities:', df['instabilities'].min(), df['instabilities'].max(),
      df['instabilities'].median())
print('max. dist. th-b:', df['max. dist th-bi'].min(),
      df['max. dist th-bi'].max(), df['max. dist th-bi'].median())
print('blasts per ep.:', df['blasts per breakthrough'].min(),
      df['blasts per breakthrough'].max(),
      df['blasts per breakthrough'].median())

# histograms for statistics of a tested agent
pltr.test_stats_histograms(df, r'06_results\histograms.jpg')

# boxplot of the actions that the agent took
pltr.test_stats_boxplot(df, r'06_results\boxplot.jpg')
