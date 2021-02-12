# -*- coding: utf-8 -*-
"""
Reinforcement Learning based Process Optimization and Strategy Development in
Conventional Tunneling; G.H. Erharter, T.F. Hansen, Z. Liu, T. Marcher
more publication info...

Plotting functionalities for either stand-alone plots in the main code, or for
rendering whole episodes. Also contains functions that plot special figures
for the paper.

Created on Wed Jul  1 15:30:52 2020
code contributors: Georg H. Erharter, Tom F. Hansen
"""

import cv2
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
from pathlib import Path
import pandas as pd
from typing import Iterable

# test

class plotter():

    def __init__(self):
        # custom colors for some of the plots
        self.garnet_red = np.array([139, 53, 56]) / 255
        self.omphacite_green = np.array([112, 126, 80]) / 255
        self.kyanite_blue = np.array([62, 78, 93]) / 255
        self.actions = ['110', '112', '150', '152', '200', '202', '220', '222']

    def custom_cmap(self, color_1: Iterable,
                    color_2: Iterable) -> LinearSegmentedColormap:
        ''' custom colormap that goes from color_1 to color_2; after:
        https://stackoverflow.com/questions/16267143/matplotlib-single-colored-colormap-with-saturation
        '''
        r1, g1, b1 = color_1
        r2, g2, b2 = color_2

        color_dict = {'red': ((0, r1, r1), (1, r2, r2)),
                      'green': ((0, g1, g1), (1, g2, g2)),
                      'blue': ((0, b1, b1), (1, b2, b2))}

        cmap = LinearSegmentedColormap('custom_cmap', color_dict)
        return cmap

    def reward_plot(self, df: pd.DataFrame, savepath: str, windows=100,
                    plot_all=True, linewidth=1.5) -> None:
        ''' plot that shows the rewards over the length of the whole training
        process as well as other logged parameters '''

        episode = int(df['episode'].iloc[-1])
        a_counts_norm = (df[self.actions].values / df['blasts per breakthrough'].values[:, None]).T

        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=5, ncols=1,
                                                      figsize=(16, 14))

        # ax1 shows a lineplot of the average reward over all episodes
        ax1.plot(df['episode'],
                 df['ep. rewards'].rolling(window=windows, center=True).mean(),
                 color=self.kyanite_blue, zorder=10)
        ax1.set_xlim(left=0, right=episode)
        ax1.set_ylim(bottom=-500)
        ax1.set_ylabel('cumulative\nepisode rewards', color=self.kyanite_blue)
        ax1.grid(alpha=0.5)
        ax1.set_title(f'{episode} episodes')

        # ax1_1 shows the development of epsilon (exploration vs. exploitation)
        ax1_1 = ax1.twinx()
        ax1_1.plot(df['episode'], df['epsilons'], color=self.kyanite_blue,
                   ls='--', lw=linewidth, zorder=10)
        ax1_1.set_xlim(left=0, right=episode)
        ax1_1.set_ylabel('-- epsilons', color=self.kyanite_blue)
        ax1_1.set_xticklabels([])

        # ax2 shows the percentage share of actions of each episode
        # add colorbar to ax2
        cax = fig.add_axes([0.955, 0.62, 0.01, 0.16])  # left, bottom, width, height
        # custom colormap
        cmap = self.custom_cmap([1, 1, 1], self.kyanite_blue)
        im = ax2.imshow(a_counts_norm, cmap=cmap, aspect='auto',
                        interpolation='none', vmax=0.5)
        fig.colorbar(im, cax=cax, orientation='vertical')
        ax2.set_xlim(left=0, right=episode)
        ax2.set_yticks(np.arange(len(self.actions)))
        ax2.set_yticklabels(self.actions)
        ax2.set_ylabel('action codes')

        # ax3 shows the target network's loss over the episodes
        ax3.plot(df['episode'], df['ep. loss'].rolling(window=windows, min_periods=1, center=True).mean(),
                 color=self.kyanite_blue, lw=linewidth, zorder=10)
        ax3.set_xlim(left=0, right=episode)
        ax3.set_ylabel('avg. loss per episode', color=self.kyanite_blue)
        ax3.set_yscale('log')
        ax3.grid(alpha=0.5)
        ax3.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))

        # ax3_1 shows the target network's accuracy over the episodes
        ax3_1 = ax3.twinx()
        ax3_1.plot(df['episode'], df['ep. accuracy'].rolling(window=windows, min_periods=1, center=True).mean(),
                   color=self.garnet_red, lw=linewidth, zorder=10)
        ax3_1.set_xlim(left=0, right=episode)
        ax3_1.set_ylabel('avg. accuracy per episode', color=self.garnet_red)
        ax3_1.set_xticklabels([])

        # ax4 shows the number of moves / blasts per episode / breakthrough
        ax4.plot(df['episode'], df['blasts per breakthrough'].rolling(window=windows, min_periods=1, center=True).mean(),
                 color=self.kyanite_blue, lw=linewidth, zorder=10)
        ax4.set_xlim(left=0, right=episode)
        ax4.grid(alpha=0.5)
        ax4.set_ylabel('avg. number of\nblasts per episode',
                       color=self.kyanite_blue)

        # ax4_1 shows the number of face instabilities per episode / breakthrough
        ax4_1 = ax4.twinx()
        ax4_1.plot(df['episode'], df['instabilities'].rolling(window=windows, min_periods=1, center=True).mean(),
                   color=self.garnet_red, lw=linewidth, zorder=10)
        ax4_1.set_xlim(left=0, right=episode)
        ax4_1.set_ylabel('avg. number of\ninstabilities per episode',
                         color=self.garnet_red)

        # computation of terminal states within last episodes for ax5 & ax5_1
        breaks = np.where(df['terminals'] == 'breakthrough', 1, np.nan)
        breaks = pd.Series(breaks).rolling(window=windows, min_periods=1,
                                           center=True).count()
        timouts = np.where(df['terminals'] == 'timeout', 1, np.nan)
        timouts = pd.Series(timouts).rolling(window=windows, min_periods=1,
                                             center=True).count()

        # ax5 shows number of breakthroughs over last episodes
        ax5.plot(df['episode'], breaks, label='breakthroughs',
                 color=self.kyanite_blue, lw=linewidth, alpha=0.9)
        ax5.set_xlim(left=0, right=episode)
        ax5.set_ylim(bottom=-5, top=windows+5)
        ax5.grid(alpha=0.5)
        ax5.set_ylabel(f'breakthroughs\nper {windows} episodes')
        ax5.set_xlabel('episodes')

        # ax5_1 shows number of time outs over last episodes
        ax5_1 = ax5.twinx()
        ax5_1.plot(df['episode'], timouts, label='timouts',
                   color=self.garnet_red, lw=linewidth, alpha=0.9)
        ax5_1.set_xlim(left=0, right=episode)
        ax5_1.set_ylim(bottom=-5, top=windows+5)
        ax5_1.set_ylabel(f'timeouts\nper {windows} episodes',
                         color=self.garnet_red)

        # optionally activate raw recorded data in the background of the axes
        if plot_all is True:
            ax1.plot(df['episode'], df['ep. rewards'],
                     color=self.kyanite_blue, alpha=0.2, zorder=0)
            ax1.set_ylim(top=500)
            ax3.plot(df['episode'], df['ep. loss'],
                     color=self.kyanite_blue, alpha=0.2, zorder=0)
            ax3.set_ylim(top=100)
            ax3_1.plot(df['episode'], df['ep. accuracy'],
                       color=self.garnet_red, alpha=0.2, zorder=0)
            ax4.plot(df['episode'], df['blasts per breakthrough'],
                     color=self.kyanite_blue, alpha=0.2, zorder=0)
            ax4.set_ylim(top=160)
            ax4_1.plot(df['episode'], df['instabilities'],
                       color=self.garnet_red, alpha=0.2, zorder=0)
            ax4_1.set_ylim(top=30)

        plt.tight_layout()
        plt.savefig(savepath, dpi=600)
        plt.close()

    def progress_plot(self, pos_ths, pos_bis, actions, rewards,
                      a1, a2, a3, a4, a5, a6, a7, a8, episode, savepath) -> None:
        ''' figure that shows the progress / selected recorded parameters of
        one selected episode '''
        n_blasts = np.arange(len(rewards))

        fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(16, 9))

        ax1.plot(n_blasts, pos_ths, color='black', zorder=2, linestyle='-',
                 linewidth=2, label='top heading')
        ax1.plot(n_blasts, pos_bis, color='black', zorder=1, linestyle='--',
                 linewidth=2, label='bench')
        ax1.fill_between(n_blasts, pos_ths - pos_bis, zorder=0,
                         color='grey',
                         label='distance: top heading - bench')
        ax1.grid(alpha=0.4)
        ax1.legend(loc='upper left', fontsize=12)

        ax1.set_xlim(left=0, right=n_blasts.max())
        ax1.set_ylabel('tunnelmeters [dm]', fontsize=12)
        ax1.set_title(f'episode: {episode+1}', fontsize=12)

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
        ax2.set_ylabel('actions', fontsize=12)
        ax2.set_yticks(np.arange(0, 8))
        ax2.set_yticklabels(['110', '112', '150', '152', '200', '202', '220',
                             '222'])

        ax3.plot(n_blasts, np.cumsum(rewards), color='black')
        ax3.set_xlim(left=0, right=n_blasts.max())
        ax3.grid(alpha=0.4)
        ax3.set_ylabel('cumulative reward', fontsize=12)
        ax3.set_xlabel('blast / move number', fontsize=12)

        plt.tight_layout()
        plt.savefig(savepath, dpi=600)
        plt.close()

    def render_geo_section(self, geo_section):
        """ funtion that visualizes the current geological section (part of the
        state) """
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
        ax.set_ylabel('geological section', fontsize=12)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.legend(geo_lines, ['not excavated', 'weak rock', 'stronger rock'],
                  loc=(1.01, 0), fontsize=12)

    def render_sup_section(self, sup_section):
        """ funtion that visualizes the current support section (part of the
        state) """
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
        """ funtion that visualizes one whole state / frame that shows the
        geological- and support section """
        fig = plt.figure(figsize=(14, 4))

        ax = fig.add_subplot(2, 1, 1)
        self.render_geo_section(geo_section)

        ax = fig.add_subplot(2, 1, 2)
        self.render_sup_section(sup_section)

        plt.tight_layout()
        plt.savefig(savepath, dpi=120)
        plt.close()

    def render_episode(self, folder, fps, x_pix, y_pix, savepath):
        """ funtion that generates a video file of one whole episode from
        multiple frames """
        n_frames = len(os.listdir(folder))

        if n_frames > 20:
            # chosen fourcc works for windows
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(savepath, fourcc, fps, (x_pix, y_pix))

            for i in range(n_frames):
                frame = cv2.imread(fr'02_plots/tmp/{i}.png')
                out.write(frame)

            out.release()

        for i in range(n_frames):
            os.remove(f'02_plots/tmp/{i}.png')

    def test_stats_histograms(self, df, savepath):
        ''' function that plots histograms for the achieved reward, number of
        face instabilities and blasts per breakthrough of a tested agent '''
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(12, 5))

        ax1.hist(df['ep. rewards'], bins=20, color=self.kyanite_blue,
                 edgecolor='black')
        ax1.set_xlabel('cumulative\nepisode reward')
        ax1.set_ylabel('n episodes')
        ax1.grid(alpha=0.5)

        ax2.hist(df['instabilities'], bins=20, color=self.kyanite_blue,
                 edgecolor='black')
        ax2.set_xlabel('face instabilities\nper episode')
        ax2.grid(alpha=0.5)

        ax3.hist(df['blasts per breakthrough'], bins=20,
                 color=self.kyanite_blue, edgecolor='black')
        ax3.set_xlabel('required blasts\nper episode')
        ax3.grid(alpha=0.5)

        plt.suptitle(f'{len(df)} test runs', y=0.98)

        plt.tight_layout(rect=(0, 0, 1, 0.95))
        plt.savefig(savepath, dpi=600)
        plt.close()

    def test_stats_boxplot(self, df, savepath):
        ''' function that creates a boxplot of the actions that a tested agent
        took '''
        action_labels = ['top.head.; al.2m; no sup.', 'top.head.; al.2m; sup.',
                         'top.head.; al.4m; no sup.', 'top.head.; al.4m, sup.',
                         'bench; al.2m; no sup.', 'bench; al.2m; sup.',
                         'bench; al.4m; no sup.', 'bench; al.4m; sup.']

        fig, ax = plt.subplots(figsize=(6, 6))

        # boxplot with whiskers that represent min-max values
        bplot = ax.boxplot(df[self.actions].values, whis=1e6,
                           patch_artist=True)

        for box, median in zip(bplot['boxes'], bplot['medians']):
            box.set(facecolor=self.kyanite_blue)
            median.set(color='black', lw=3)

        ax.set_xticklabels(action_labels, rotation=45, ha='right')
        ax.set_ylabel('n choices of an action per ep.')
        ax.set_title(f'{len(df)} test runs')
        ax.grid(alpha=0.5)

        plt.tight_layout()
        plt.savefig(savepath, dpi=600)
        plt.close()

    def multi_agent_plot(self, max_eps, rewards, instabilities,
                         n_blasts, losses, savepath, window=500):
        """ plot that allows visualization of statistics from up to 10
        agents """

        # set the rcParams to a high value as there are many datapoints to plot
        plt.rcParams['agg.path.chunksize'] = 40_000

        colors = [f'C{i}' for i in range(10)]

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1,
                                                 figsize=(14, 10))

        # plot data to the 4 axes
        for i in range(len(max_eps)):
            if i < 10:
                x = np.arange(max_eps[i])

                ax1.plot(x, rewards[i], color=colors[i], alpha=0.1)
                ax1.plot(x, rewards[i].rolling(window=window,
                                               center=True).mean(),
                         color=colors[i], label=f'agent {i+1}')

                ax2.plot(x, n_blasts[i], color=colors[i], alpha=0.1)
                ax2.plot(x, n_blasts[i].rolling(window=window,
                                                center=True).mean(),
                         color=colors[i], label=f'agent {i+1}')

                ax3.plot(x, instabilities[i], color=colors[i], alpha=0.1)
                ax3.plot(x, instabilities[i].rolling(window=window,
                                                     center=True).mean(),
                         color=colors[i], label=f'agent {i+1}')

                ax4.plot(x, losses[i], color=colors[i], alpha=0.1)
                ax4.plot(x, losses[i].rolling(window=window,
                                              center=True).mean(),
                         color=colors[i], label=f'agent {i+1}')

        # individual styling of axes
        ax1.set_xlim(left=0, right=max(max_eps))
        ax1.set_ylim(top=450, bottom=-300)
        ax1.set_ylabel('cumulative\nepisode rewards')
        ax1.grid(alpha=0.5)
        ax1.legend(loc='lower right')

        ax2.set_xlim(left=0, right=max(max_eps))
        ax2.set_ylim(top=180, bottom=100)
        ax2.set_ylabel('avg. number of\nblasts per episode')
        ax2.grid(alpha=0.5)
        ax2.legend(loc='upper right')

        ax3.set_ylim(top=25, bottom=0)
        ax3.set_xlim(left=0, right=max(max_eps))
        ax3.set_ylabel('avg. number of\ninstabilities per episode')
        ax3.grid(alpha=0.5)
        ax3.legend(loc='upper right')

        ax4.set_xlim(left=0, right=max(max_eps))
        # ax4.set_ylim(top=100)
        ax4.set_yscale('log')
        ax4.set_ylabel('avg. loss per episode')
        ax4.set_xlabel('episodes')
        ax4.grid(alpha=0.5)
        ax4.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
        ax4.legend(loc='upper right')

        plt.tight_layout()
        plt.savefig(savepath, dpi=600)
        plt.close()


if __name__ == '__main__':

    # plot selected training progress plot

    plt.rcParams['agg.path.chunksize'] = 40_000

    pltr = plotter()

    df = pd.read_csv(r'02_plots\2020_10_10\episode_stats.csv')

    pltr.reward_plot(df, savepath=Path(r'06_results\2020_10_10_training.png'),
                     windows=500, linewidth=2, plot_all=True)
