# -*- coding: utf-8 -*-
"""
Created on Mon Dec 6, 2021

@author: abrielmann

"""
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

# import and update global figure settings
import matplotlib.pylab as pylab
params = {'legend.fontsize': 10,
          'legend.title_fontsize': 12,
          'legend.borderpad': 0,
          'figure.figsize': (8,10),
         'axes.labelsize': 10,
         'axes.titlesize': 12,
         'xtick.labelsize': 10,
         'ytick.labelsize': 10,
         'lines.linewidth': 2,
         'image.cmap': 'gray',
         'savefig.dpi': 300}
pylab.rcParams.update(params)

#%% directories
os.chdir('..')
homeDir = os.getcwd()
dataDir = homeDir + '/exp3_prolificShort'

#%% fetch pre-processed data
rawDf = pd.read_csv(dataDir + '/all_data.csv')
resDfPrecue = pd.read_csv(dataDir + '/res_one_Precue.csv')
resDfPrecue['cue'] = 'Pre'
resDfPrecue['task'] = 'rateOne'
resDfPostcue = pd.read_csv(dataDir + '/res_one_Postcue.csv')
resDfPostcue['cue'] = 'Post'
resDfPostcue['task'] = 'rateOne'

resDfPrecueBoth = pd.read_csv(dataDir + '/res_both_Precue.csv')
resDfPrecueBoth['cue'] = 'Pre'
resDfPrecueBoth['task'] = 'rateBoth'
resDfPostcueBoth = pd.read_csv(dataDir + '/res_both_Postcue.csv')
resDfPostcueBoth['cue'] = 'Post'
resDfPostcueBoth['task'] = 'rateBoth'

# merge result DFs
resDf = pd.concat([resDfPrecue, resDfPostcue, resDfPrecueBoth, resDfPostcueBoth])
rmseDf = pd.wide_to_long(resDf.reset_index(), ['avgRMSE'],
                         i=['participant','cue', 'task'],
                         j='model', sep='_', suffix=r'\w+')
rmseDf.reset_index(inplace=True)

# get cronbach's; corrs
relDf = pd.read_csv(dataDir + '/reliabilities_correlations.csv')

# manual entry of reported alphas and correlations in Brielmann & Pelli (2020)
conditions = ['one precued', 'one postcued', 'both precued', 'both postcued']
alpha_2images = [0.92, 0.89, 0.85, 0.81]
relations = ['one pre vs post', 'both pre vs post', 'pre one vs both',
             'post one vs both']
corrs_2images = [0.93, 0.92, 0.52, 0.45]

#%% plot
# use gridspec to define layout
fig = plt.figure()
gs = fig.add_gridspec(ncols=2, nrows=3)

ax0 = fig.add_subplot(gs[0, 0]) # alphas
ax1 = fig.add_subplot(gs[0, 1]) # corrs
ax2 = fig.add_subplot(gs[1, :]) # RMSEs rate one
ax3 = fig.add_subplot(gs[2, :]) # RMSEs rate both

# alphas
ax0.plot(np.arange(4), alpha_2images, 'ok', label='Brielmann & Pelli (2020)')
ax0.errorbar(np.arange(4), relDf.alpha,
             yerr = [relDf.alpha_upperCI-relDf.alpha, relDf.alpha-relDf.alpha_lowerCI],
             fmt='or',
             label='current study')
ax0.set_ylabel('Cronbach\'s ' + r'$\alpha$')
ax0.legend(loc='lower right', frameon=False)
ax0.set_xticks([0, 0.5, 1, 2, 2.5, 3])
ax0.set_xticklabels(['precue', '\n one-pleasure', 'postcue',
                     'precue', '\n combined-pleasure', 'postcue'])
ax0.text(-.2, 1.1, 'A', transform=ax0.transAxes,
  fontsize=12, fontweight='bold', va='top', ha='right')

# correlations
ax1.plot(np.arange(4), corrs_2images, 'ok', label='Brielmann & Pelli (2020)')
ax1.errorbar(np.arange(4), relDf.corrs,
             yerr = [relDf.corr_upperCI-relDf.corrs, relDf.corrs-relDf.corr_lowerCI],
             fmt='or',
             label='current study')
ax1.set_ylabel('Pearson\'s ' + '$r$')
ax1.set_xticks([0, 0.5, 1, 2, 2.5, 3])
ax1.set_xticklabels(['precue', '\n one-pleasure', 'postcue',
                     'precue', '\n combined-pleasure', 'postcue'])
ax1.text(-.2, 1.1, 'B', transform=ax1.transAxes,
  fontsize=12, fontweight='bold', va='top', ha='right')

# RMSEs for one-pleasure
sns.boxplot(data=rmseDf[rmseDf.task=='rateOne'], x='model', y='avgRMSE',
            hue='cue', ax=ax2,
            order=['accurate', 'linear', 'averaging', 'linear_modality'])
ax2.legend(loc='upper left', frameon=False)
ax2.set_ylim((0,4))
ax2.set_xticklabels(['faithful', 'linear', 'averaging', 'linear modality'])
ax2.set_title('one-pleasure trials')
ax2.set_ylabel('mean RMSE')
ax2.text(-.075, 1.1, 'C', transform=ax2.transAxes,
  fontsize=12, fontweight='bold', va='top', ha='right')

# RMSEs for combined-pleasure
sns.boxplot(data=rmseDf[rmseDf.task=='rateBoth'], x='model', y='avgRMSE',
            hue='cue', ax=ax3,
            order=['averaging', 'compress', 'extreme', 'linear_modality'])
ax3.legend(loc='upper left', frameon=False)
ax3.set_ylim((0,4))
ax3.set_xticklabels(['averaging', 'compressive', 'expansive', 'linear modality'])
ax3.set_title('combined-pleasure trials')
ax3.set_ylabel('mean RMSE')
ax3.text(-.075, 1.1, 'D', transform=ax3.transAxes,
  fontsize=12, fontweight='bold', va='top', ha='right')


sns.despine()
plt.subplots_adjust(hspace=.67, wspace=.5)
plt.show()
plt.close()