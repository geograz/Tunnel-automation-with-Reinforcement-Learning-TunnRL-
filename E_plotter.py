# -*- coding: utf-8 -*-
"""
Reinforcement Learning based Process Optimization and Strategy Development in
Conventional Tunneling; G.H. Erharter, T.F. Hansen, Z. Liu, T. Marcher
more publication info...

Plotting functionalities for either stand-alone plots in the main code, or for
rendering whole episodes. Also contains functions that plot special figures
for the paper.

Created on Wed Jul  1 15:30:52 2020
code contributors: G.H. Erharter
"""

import cv2
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
import pandas as pd


class plotter():

    def __init__(self):
        # custom colors for all plots
        self.garnet_red = np.array([139, 53, 56]) / 255
        self.omphacite_green = np.array([112, 126, 80]) / 255
        self.kyanite_blue = np.array([62, 78, 93]) / 255

    def custom_cmap(self, color1, color2):
        # https://stackoverflow.com/questions/16267143/matplotlib-single-colored-colormap-with-saturation
        # from color r,g,b
        r1, g1, b1 = color1

        # to color r,g,b
        r2, g2, b2 = color2

        cdict = {'red': ((0, r1, r1),
                         (1, r2, r2)),
                 'green': ((0, g1, g1),
                           (1, g2, g2)),
                 'blue': ((0, b1, b1),
                          (1, b2, b2))}

        cmap = LinearSegmentedColormap('custom_cmap', cdict)
        return cmap

    def reward_plot(self, df, savepath, windows=100, plot_eprewpoints=True,
                    linewidth=1.5):

        episode = int(df['episode'].iloc[-1])
        actions = ['110', '112', '150', '152', '200', '202', '220', '222']
        a_counts_norm = (df[actions].values / df['blasts per breakthrough'].values[:, None]).T

        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=5, ncols=1,
                                                      figsize=(16, 14))
        if plot_eprewpoints is True:
            ax1.scatter(df['episode'], df['ep. rewards'],
                        color=self.kyanite_blue, s=0.5, alpha=0.1)
        ax1.plot(df['episode'],
                 df['ep. rewards'].rolling(window=windows, center=True).mean(),
                 color=self.kyanite_blue,)
        ax1.set_xlim(left=0, right=episode)
        ax1.set_ylim(bottom=-500)
        ax1.set_ylabel('cumulative\nepisode rewards', color=self.kyanite_blue)
        ax1.grid(alpha=0.5)
        ax1.set_title(f'{episode} episodes')

        ax1_1 = ax1.twinx()
        ax1_1.plot(df['episode'], df['epsilons'], color=self.kyanite_blue,
                   ls='--', lw=linewidth)
        ax1_1.set_xlim(left=0, right=episode)
        ax1_1.set_ylabel('-- epsilons', color=self.kyanite_blue)
        ax1_1.set_xticklabels([])

        # add colorbar to ax2
        cax = fig.add_axes([0.955, 0.62, 0.01, 0.16])  # left, bottom, width, height
        # custom colormap
        cmap = self.custom_cmap([1, 1, 1], self.kyanite_blue)
        im = ax2.imshow(a_counts_norm, cmap=cmap, aspect='auto',
                        interpolation='none', vmax=0.5)
        fig.colorbar(im, cax=cax, orientation='vertical')
        ax2.set_xlim(left=0, right=episode)
        ax2.set_yticks(np.arange(len(actions)))
        ax2.set_yticklabels(actions)
        ax2.set_ylabel('action codes')

        ax3.plot(df['episode'], df['ep. loss'].rolling(window=windows, min_periods=1, center=True).mean(),
                 color=self.kyanite_blue, lw=linewidth, alpha=0.9)
        ax3.set_xlim(left=0, right=episode)
        ax3.set_ylabel('avg. loss per episode', color=self.kyanite_blue)
        ax3.set_yscale('log')
        ax3.grid(alpha=0.5)
        ax3.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))

        ax3_1 = ax3.twinx()
        ax3_1.plot(df['episode'], df['ep. accuracy'].rolling(window=windows, min_periods=1, center=True).mean(),
                   color=self.garnet_red, lw=linewidth, alpha=0.9)
        ax3_1.set_xlim(left=0, right=episode)
        ax3_1.set_ylabel('avg. accuracy per episode', color=self.garnet_red)
        ax3_1.set_xticklabels([])

        ax4.plot(df['episode'], df['blasts per breakthrough'].rolling(window=windows, min_periods=1, center=True).mean(),
                 color=self.kyanite_blue, lw=linewidth, alpha=0.9)
        ax4.set_xlim(left=0, right=episode)
        ax4.grid(alpha=0.5)
        ax4.set_ylabel('avg. number of\nblasts per episode',
                       color=self.kyanite_blue)

        ax4_1 = ax4.twinx()
        ax4_1.plot(df['episode'], df['instabilities'].rolling(window=windows, min_periods=1, center=True).mean(),
                   color=self.garnet_red, lw=linewidth, alpha=0.9)
        ax4_1.set_xlim(left=0, right=episode)
        ax4_1.set_ylabel('avg. number of\ninstabilities per episode',
                         color=self.garnet_red)

        breaks = np.where(df['terminals'] == 'breakthrough', 1, np.nan)
        breaks = pd.Series(breaks).rolling(window=windows, min_periods=1,
                                           center=True).count()
        timouts = np.where(df['terminals'] == 'timeout', 1, np.nan)
        timouts = pd.Series(timouts).rolling(window=windows, min_periods=1,
                                             center=True).count()

        ax5.plot(df['episode'], breaks, label='breakthroughs',
                 color=self.kyanite_blue, lw=linewidth, alpha=0.9)
        ax5.set_xlim(left=0, right=episode)
        ax5.set_ylim(bottom=-5, top=windows+5)
        ax5.grid(alpha=0.5)
        ax5.set_ylabel(f'breakthroughs\nper {windows} episodes')
        ax5.set_xlabel('episodes')

        ax5_1 = ax5.twinx()
        ax5_1.plot(df['episode'], timouts, label='timouts',
                   color=self.garnet_red, lw=linewidth, alpha=0.9)
        ax5_1.set_xlim(left=0, right=episode)
        ax5_1.set_ylim(bottom=-5, top=windows+5)
        ax5_1.set_ylabel(f'timeouts\nper {windows} episodes',
                         color=self.garnet_red)

        plt.tight_layout()
        plt.savefig(savepath, dpi=600)
        plt.close()

    def progress_plot(self, pos_ths, pos_bis, actions, rewards,
                      a1, a2, a3, a4, a5, a6, a7, a8, episode, savepath):

        n_blasts = np.arange(len(rewards))

        fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(20, 8))

        ax1.plot(n_blasts, pos_ths, color='black', zorder=2, linestyle='-',
                 linewidth=2, label='top heading')
        ax1.plot(n_blasts, pos_bis, color='black', zorder=1, linestyle='--',
                 linewidth=2, label='bench')
        ax1.fill_between(n_blasts, pos_ths - pos_bis, zorder=0,
                         color='grey',
                         label='distance: top heading - bench')
        ax1.grid(alpha=0.4)
        ax1.legend(loc='upper left')

        ax1.set_xlim(left=0, right=n_blasts.max())
        ax1.set_ylabel('tunnelmeters [dm]')
        ax1.set_title(f'episode: {episode}')

        idx = np.where(actions == a1)[0]-1  # -1 because decision is made before blast
        ax2.scatter(n_blasts[idx], np.full(len(n_blasts), 0)[idx],
                    color='grey', edgecolor='black')
        idx = np.where(actions == a2)[0]-1
        ax2.scatter(n_blasts[idx], np.full(len(n_blasts), 1)[idx],
                    color='grey', edgecolor='black')
        idx = np.where(actions == a3)[0]-1
        ax2.scatter(n_blasts[idx], np.full(len(n_blasts), 2)[idx],
                    color='grey', edgecolor='black')
        idx = np.where(actions == a4)[0]-1
        ax2.scatter(n_blasts[idx], np.full(len(n_blasts), 3)[idx],
                    color='grey', edgecolor='black')
        idx = np.where(actions == a5)[0]-1
        ax2.scatter(n_blasts[idx], np.full(len(n_blasts), 4)[idx],
                    color='grey', edgecolor='black')
        idx = np.where(actions == a6)[0]-1
        ax2.scatter(n_blasts[idx], np.full(len(n_blasts), 5)[idx],
                    color='grey', edgecolor='black')
        idx = np.where(actions == a7)[0]-1
        ax2.scatter(n_blasts[idx], np.full(len(n_blasts), 6)[idx],
                    color='grey', edgecolor='black')
        idx = np.where(actions == a8)[0]-1
        ax2.scatter(n_blasts[idx], np.full(len(n_blasts), 7)[idx],
                    color='grey', edgecolor='black')

        ax2.grid(axis='y', alpha=0.4)
        ax2.set_xlim(left=0, right=n_blasts.max())
        ax2.set_ylabel('actions')
        ax2.set_yticks(np.arange(0, 8))
        ax2.set_yticklabels(['110', '112', '150', '152', '200', '202', '220',
                             '222'])

        ax3.plot(n_blasts, np.cumsum(rewards), color='black')
        ax3.set_xlim(left=0, right=n_blasts.max())
        ax3.grid(alpha=0.4)
        ax3.set_ylabel('cumulative reward')
        ax3.set_xlabel('number of blasts')

        ax3_1 = ax3.twinx()
        ax3_1.plot(n_blasts, rewards, color='black', ls='--')
        ax3_1.set_xlim(left=0, right=n_blasts.max())
        ax3_1.set_yscale('log')
        ax3_1.set_ylabel('rewards')

        plt.tight_layout()
        plt.savefig(savepath, dpi=300)
        plt.close()

    def render_geo_section(self, geo_section):
        # dictionary with colors for rockmass
        geo_colors = {0: np.array([0, 0, 0]),  # not excavated yet
                      1: np.array([185, 122, 87]),  # brown = weak rock
                      2: np.array([112, 146, 190])}  # blue = strong rock

        # create 3D RGB frames
        geo_frame = np.repeat(geo_section[:, :, np.newaxis], 3, axis=2)

        # replace with RGB values
        for i in range(3):
            geo_frame[:, :, i] = np.where(geo_frame[:, :, i] == 1,
                                          geo_colors[1][i], geo_frame[:, :, i])
            geo_frame[:, :, i] = np.where(geo_frame[:, :, i] == 2,
                                          geo_colors[2][i], geo_frame[:, :, i])

        geo_frame = geo_frame/255

        # make custom legends
        geo_lines = [Line2D([0], [0], color=geo_colors[0]/255, lw=4),
                     Line2D([0], [0], color=geo_colors[1]/255, lw=4),
                     Line2D([0], [0], color=geo_colors[2]/255, lw=4)]

        ax = plt.gca()
        ax.imshow(geo_frame, aspect='auto', interpolation='none')
        ax.set_ylabel('geological section')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.legend(geo_lines, ['not excavated', 'weak rock', 'stronger rock'],
                  loc=(1.01, 0), fontsize=8)

    def render_sup_section(self, sup_section):
        # dictionary with colors for support
        sup_colors = {0: np.array([0, 0, 0]), 1: np.array([195, 195, 195])}

        # create 3D RGB frames
        sup_frame = np.repeat(sup_section[:, :, np.newaxis], 3, axis=2)

        # replace with RGB values
        for i in range(3):
            sup_frame[:, :, i] = np.where(sup_frame[:, :, i] == 1,
                                          sup_colors[1][i], sup_frame[:, :, i])

        sup_frame = sup_frame/255

        # make custom legends
        sup_lines = [Line2D([0], [0], color=sup_colors[0]/255, lw=4),
                     Line2D([0], [0], color=sup_colors[1]/255, lw=4)]

        ax = plt.gca()
        ax.imshow(sup_frame, aspect='auto', interpolation='none')
        ax.set_ylabel('installed facesupport')
        ax.set_xlabel('tunnellength [dm]')
        ax.set_yticklabels([])
        ax.legend(sup_lines, ['no support installed', 'support installed'],
                  loc=(1.01, 0), fontsize=8)

    def render_frame(self, geo_section, sup_section, savepath):

        fig = plt.figure(figsize=(14, 4))

        ax = fig.add_subplot(2, 1, 1)
        self.render_geo_section(geo_section)

        ax = fig.add_subplot(2, 1, 2)
        self.render_sup_section(sup_section)

        plt.tight_layout()
        plt.savefig(savepath, dpi=120)
        plt.close()

    def render_episode(self, folder, fps, x_pix, y_pix, savepath):
        n_frames = len(os.listdir(folder))

        if n_frames > 20:
            # windows:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(savepath, fourcc, fps, (x_pix, y_pix))

            for i in range(n_frames):
                frame = cv2.imread(fr'02_plots\tmp\{i}.png')
                out.write(frame)

            out.release()

        for i in range(n_frames):
            os.remove(fr'02_plots\tmp\{i}.png')
