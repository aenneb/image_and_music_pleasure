# -*- coding: utf-8 -*-
"""
Created on Wed Dec 8 2021

@author: abrielmann

"""
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns

# import and update global figure settings
import matplotlib.pylab as pylab
params = {'legend.fontsize': 10,
          'legend.title_fontsize': 12,
          'legend.borderpad': 0,
          'figure.figsize': (12,12),
         'axes.labelsize': 10,
         'axes.titlesize': 12,
         'xtick.labelsize': 10,
         'ytick.labelsize': 10,
         'lines.linewidth': 2,
         'image.cmap': 'gray',
         'savefig.dpi': 300}
pylab.rcParams.update(params)

os.chdir('..')
os.chdir('..')
homeDir = os.getcwd()
dataDir = homeDir + '/exp3_prolificShort'

#%% fetch pre-processed data
rawDf = pd.read_csv(dataDir + '/all_data.csv')
resDfPrecue = pd.read_csv(dataDir + '/res_both_Precue.csv')
resDfPrecue['cue'] = 'Pre'
resDfPostcue = pd.read_csv(dataDir + '/res_both_Postcue.csv')
resDfPostcue['cue'] = 'Post'

# get averages from data
preOneDf = rawDf.loc[(rawDf['cueTime']=='Pre') & (rawDf['cued']=='Both')]
postOneDf = rawDf.loc[(rawDf['cueTime']=='Post') & (rawDf['cued']=='Both')]
avgsPreOne = preOneDf.groupby(['baselineMusic', 'baselineImage'])['rating'].mean().unstack()
avgsPostOne = postOneDf.groupby(['baselineMusic', 'baselineImage'])['rating'].mean().unstack()

# merge result DFs
resDf = pd.concat([resDfPrecue, resDfPostcue])
rmseDf = pd.wide_to_long(resDf, ['avgRMSE'], i=['participant','cue'],
                         j='model', sep='_', suffix=r'\w+')
rmseDf.reset_index(inplace=True)

# get grand average of model fits
grandAvgRes_a_compress = resDf['avgRes_a_compress'].mean()
grandAvgRes_a_compress_pre = resDf.loc[resDf['cue']=='Pre','avgRes_a_compress'].mean()
grandAvgRes_a_compress_post = resDf.loc[resDf['cue']=='Post','avgRes_a_compress'].mean()
grandAvgRes_b_compress = resDf['avgRes_b_compress'].mean()
grandAvgRes_b_compress_pre = resDf.loc[resDf['cue']=='Pre','avgRes_b_compress'].mean()
grandAvgRes_b_compress_post = resDf.loc[resDf['cue']=='Post','avgRes_b_compress'].mean()

grandAvgRes_a_extreme = resDf['avgRes_a_extreme'].mean()
grandAvgRes_a_extreme_pre = resDf.loc[resDf['cue']=='Pre','avgRes_a_extreme'].mean()
grandAvgRes_a_extreme_post = resDf.loc[resDf['cue']=='Post','avgRes_a_extreme'].mean()
grandAvgRes_b_extreme = resDf['avgRes_b_extreme'].mean()
grandAvgRes_b_extreme_pre = resDf.loc[resDf['cue']=='Pre','avgRes_b_extreme'].mean()
grandAvgRes_b_extreme_post = resDf.loc[resDf['cue']=='Post','avgRes_b_extreme'].mean()

#%% generate predictions
def pred_linear(weight, targetPleasure, distractorPleasure):
    predictions = weight*targetPleasure + (1-weight)*distractorPleasure
    return predictions

def pred_linearTransform(a, b, targetPleasure, distractorPleasure):
    predictions = a + b*( 0.5*targetPleasure + 0.5*distractorPleasure)
    return predictions

imagePleasure = np.repeat(np.arange(1,10), 9)
musicPleasure = np.tile(np.arange(1,10), 9)

predCompress = pred_linearTransform(grandAvgRes_a_compress,
                                    grandAvgRes_b_compress,
                                    imagePleasure, musicPleasure)
predCompressPre = pred_linearTransform(grandAvgRes_a_compress_pre,
                                    grandAvgRes_b_compress_pre,
                                       imagePleasure, musicPleasure)
predCompressPost = pred_linearTransform(grandAvgRes_a_compress_post,
                                    grandAvgRes_b_compress_post,
                                        imagePleasure, musicPleasure)

predExtreme = pred_linearTransform(grandAvgRes_a_extreme,
                                    grandAvgRes_b_extreme,
                                    imagePleasure, musicPleasure)
predExtremePre = pred_linearTransform(grandAvgRes_a_extreme_pre,
                                    grandAvgRes_b_extreme_pre,
                                       imagePleasure, musicPleasure)
predExtremePost = pred_linearTransform(grandAvgRes_a_extreme_post,
                                    grandAvgRes_b_extreme_post,
                                        imagePleasure, musicPleasure)

predAveraging = pred_linear(0.5, imagePleasure, musicPleasure)


predAveragingDf = pd.DataFrame({'imagePleasure': imagePleasure,
                             'musicPleasure': musicPleasure,
                            'predictions': predAveraging})
predAveragingDf = predAveragingDf.pivot(index='imagePleasure',
                                  columns='musicPleasure',
                                  values='predictions')

predCompressDf = pd.DataFrame({'imagePleasure': imagePleasure,
                             'musicPleasure': musicPleasure,
                            'predictions': predCompress})
predCompressDf = predCompressDf.pivot(index='imagePleasure',
                                  columns='musicPleasure',
                                  values='predictions')

predExtremeDf = pd.DataFrame({'imagePleasure': imagePleasure,
                             'musicPleasure': musicPleasure,
                            'predictions': predExtreme})
predExtremeDf = predExtremeDf.pivot(index='imagePleasure',
                                  columns='musicPleasure',
                                  values='predictions')

#%% print average RMSEs
rmseDf.groupby(['cue','model']).mean()

#%% Re-create those plots
# use gridspec to define layout
fig = plt.figure()
gs = fig.add_gridspec(4,4)

# inverted U
ax0 = fig.add_subplot(gs[0, 0]) # faithful model
ax1 = fig.add_subplot(gs[0, 1]) # partial averaging model
ax2 = fig.add_subplot(gs[0, 2]) # for colorbar/empty
cax = inset_axes(ax2, width="10%", height="100%", loc='lower left', borderpad=0)

ax3 = fig.add_subplot(gs[1, 0]) # averaging model
ax4 = fig.add_subplot(gs[1, 1]) # data pre-cued
ax5 = fig.add_subplot(gs[1, 2]) # data post-cued

ax6 = fig.add_subplot(gs[2, 0]) # faithful model pre-cue
ax7 = fig.add_subplot(gs[2, 1]) # partial averaging model pre-cue
ax8 = fig.add_subplot(gs[2, 2]) # averaging model pre-cue
ax9 = fig.add_subplot(gs[3, 0]) # faithful model post-cue
ax10 = fig.add_subplot(gs[3, 1]) # partial averaging model post-cue
ax11 = fig.add_subplot(gs[3, 2]) # averaging model post-cue


sns.heatmap(avgsPreOne, cmap='viridis', vmin=1, vmax=9, ax=ax0, cbar=False)
ax0.set_title('average data [precue]')
sns.heatmap(avgsPostOne, cmap='viridis', vmin=1, vmax=9, ax=ax1,
            cbar=True, cbar_ax=cax)
ax1.collections[0].colorbar.set_label('mean rating')
ax1.set_title('average data [postcue]')
ax2.set_axis_off()

# Now that we have all sub-plots placed and assigned, let's plot!
sns.heatmap(predAveragingDf.T, cmap='viridis', vmin=1, vmax=9, ax=ax3, cbar=False)
ax3.set_title('Faithful averaging')
sns.heatmap(predCompressDf.T, cmap='viridis', vmin=1, vmax=9, ax=ax4, cbar=False)
ax4.set_title('Compressive')
sns.heatmap(predExtremeDf.T, cmap='viridis', vmin=1, vmax=9, ax=ax5, cbar=False)
ax5.set_title('Expansive')

# Now, data vs. prediction for pre- and post-cued blocks separetely
sns.lineplot(data=rawDf.loc[rawDf['cueTime']=='Pre'], x='baselineImage',
              y='rating', marker='o', linestyle='', err_style='bars',
              label='data', ax=ax6)
sns.lineplot(x=imagePleasure, y=(imagePleasure+musicPleasure)/2,
             marker='o', linestyle='', err_style='bars',
             label='model', ax=ax6)
ax6.plot([1,9], [1,9], '--k')

sns.lineplot(data=rawDf.loc[rawDf['cueTime']=='Pre'], x='baselineImage',
              y='rating', marker='o', linestyle='', err_style='bars',
              label='data', ax=ax7)
sns.lineplot(x=imagePleasure, y=predCompressPre,
              marker='o', linestyle='', err_style='bars',
              label='model', ax=ax7)
ax7.plot([1,9], [1,9], '--k')
ax7.set_title('precue')

sns.lineplot(data=rawDf.loc[rawDf['cueTime']=='Pre'], x='baselineImage',
              y='rating', marker='o', linestyle='', err_style='bars',
              label='data', ax=ax8)
sns.lineplot(x=imagePleasure, y=predExtremePre,
              marker='o', linestyle='', err_style='bars',
              label='model', ax=ax8)
ax8.plot([1,9], [1,9], '--k')

sns.lineplot(data=rawDf.loc[rawDf['cueTime']=='Post'], x='baselineImage',
              y='rating', marker='o', linestyle='', err_style='bars',
              label='data', ax=ax9)
sns.lineplot(x=imagePleasure, y=(imagePleasure+musicPleasure)/2,
             marker='o', linestyle='', err_style='bars',
             label='model', ax=ax9)
ax9.plot([1,9], [1,9], '--k')

sns.lineplot(data=rawDf.loc[rawDf['cueTime']=='Post'], x='baselineImage',
              y='rating', marker='o', linestyle='', err_style='bars',
              label='data', ax=ax10)
sns.lineplot(x=imagePleasure, y=predCompressPost,
              marker='o', linestyle='', err_style='bars',
              label='model', ax=ax10)
ax10.plot([1,9], [1,9], '--k')
ax10.set_title('postcue')

sns.lineplot(data=rawDf.loc[rawDf['cueTime']=='Post'], x='baselineImage',
              y='rating', marker='o', linestyle='', err_style='bars',
              label='data', ax=ax11)
sns.lineplot(x=imagePleasure, y=predExtremePost,
              marker='o', linestyle='', err_style='bars',
              label='model', ax=ax11)
ax11.plot([1,9], [1,9], '--k')

sns.despine()
plt.subplots_adjust(hspace=.5, wspace=.5)
plt.show()
plt.close()
