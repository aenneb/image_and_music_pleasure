# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 12:26:16 2021

@author: abrielmann
"""
import os
import numpy as np
import pandas as pd
import pingouin as pg # for cronbach's alpha; see https://pingouin-stats.org/generated/pingouin.cronbach_alpha.html
from matplotlib import pyplot as plt

#%% set directories
os.chdir('..')
homeDir = os.getcwd()
dataDir = homeDir + '/exp3_prolificShort'

#%% fetch pre-processed data
df = pd.read_csv(dataDir + '/all_data.csv')
participants = np.unique(df.participant)
nParticipants = len(participants)
images = np.unique(df.image)
music = np.unique(df.music)
stims = np.concatenate((images, music))
nImages = len(images)
nSongs = len(music)
nStims = nImages+nSongs

#%% Loop through participants and get 'errors', i.e., deviation of rating
# compared to baseline per image per cueing condition

err_pre_one = np.empty((nStims, nParticipants))
err_post_one = np.empty((nStims, nParticipants))
err_pre_both = np.empty((nStims, nParticipants))
err_post_both = np.empty((nStims, nParticipants))

countPeeps = 0
for peep in participants:
    peepsDf = df.loc[df['participant']==peep]

    countStims = 0
    for stim in stims:
        stimInd = (peepsDf['cuedImage']==stim) | (peepsDf['cuedMusic']==stim)
        thisDf = peepsDf.loc[stimInd]
        precueDf = thisDf.loc[thisDf['cueTime']=='Pre']
        postcueDf = thisDf.loc[thisDf['cueTime']=='Post']
        err_pre_one[countStims, countPeeps] = precueDf.loc[precueDf['cued']!='Both', 'diffRatePred']
        err_post_one[countStims, countPeeps] = postcueDf.loc[postcueDf['cued']!='Both', 'diffRatePred']

        err_pre_both[countStims, countPeeps] = np.mean(np.abs(precueDf.loc[precueDf['cued']=='Both', 'diffRatePred']))
        err_post_both[countStims, countPeeps] = np.mean(np.abs(postcueDf.loc[postcueDf['cued']=='Both', 'diffRatePred']))

        countStims += 1

    countPeeps += 1

# %% With these 'errors' calculated, we can now compute cronbach's alpha for
# each condition
alpha_pre_one, alpha_pre_one_CI = pg.cronbach_alpha(pd.DataFrame(err_pre_one.T))
alpha_post_one, alpha_post_one_CI = pg.cronbach_alpha(pd.DataFrame(err_post_one.T))

alpha_pre_both, alpha_pre_both_CI = pg.cronbach_alpha(pd.DataFrame(err_pre_both.T))
alpha_post_both, alpha_post_both_CI = pg.cronbach_alpha(pd.DataFrame(err_post_both.T))

alphas = [alpha_pre_one, alpha_post_one, alpha_pre_both, alpha_post_both]
alpha_negErr = [alpha_pre_one-alpha_pre_one_CI[0],
              alpha_post_one-alpha_post_one_CI[0],
              alpha_pre_both-alpha_pre_both_CI[0],
              alpha_post_both-alpha_post_both_CI[0]]
alpha_posErr = [alpha_pre_one_CI[1]-alpha_pre_one,
              alpha_post_one_CI[1]-alpha_post_one,
              alpha_pre_both_CI[1]-alpha_pre_both,
              alpha_post_both_CI[1]-alpha_post_both]
alpha_lowerCIs = [alpha_pre_one_CI[0],
              alpha_post_one_CI[0],
              alpha_pre_both_CI[0],
              alpha_post_both_CI[0]]
alpha_upperCIs = [alpha_pre_one_CI[1],
              alpha_post_one_CI[1],
              alpha_pre_both_CI[1],
              alpha_post_both_CI[1]]

#%% the alphas above also provide an upper bound to possible correlations
# between errors across trial types
max_r_prePost_one = np.sqrt(alpha_pre_one*alpha_post_one)
max_r_prePost_both = np.sqrt(alpha_pre_both*alpha_post_both)
max_r_oneBoth_pre = np.sqrt(alpha_pre_one*alpha_pre_both)
max_r_oneBoth_post = np.sqrt(alpha_post_one*alpha_post_both)

#%% Now calculate the actual correlations
corr_prePost_one = pg.corr(err_pre_one.flatten(), err_post_one.flatten())
corr_prePost_both = pg.corr(err_pre_both.flatten(), err_post_both.flatten())
corr_oneBoth_pre = pg.corr(err_pre_one.flatten(), err_pre_both.flatten())
corr_oneBoth_post = pg.corr(err_post_one.flatten(), err_post_both.flatten())

corrs = [corr_prePost_one.r.values[0], corr_prePost_both.r.values[0],
         corr_oneBoth_pre.r.values[0], corr_oneBoth_post.r.values[0]]
corrs = [l.tolist() for l in corrs]
corr_negErr = [corr_prePost_one.r.values[0] - corr_prePost_one['CI95%'][0][0],
               corr_prePost_both.r.values[0] - corr_prePost_both['CI95%'][0][0],
               corr_oneBoth_pre.r.values[0] - corr_oneBoth_pre['CI95%'][0][0],
               corr_oneBoth_post.r.values[0] - corr_oneBoth_post['CI95%'][0][0]]
corr_posErr = [corr_prePost_one['CI95%'][0][1] - corr_prePost_one.r.values[0],
               corr_prePost_both['CI95%'][0][1] - corr_prePost_both.r.values[0],
               corr_oneBoth_pre['CI95%'][0][1] - corr_oneBoth_pre.r.values[0],
               corr_oneBoth_post['CI95%'][0][1] - corr_oneBoth_post.r.values[0]]

corr_lowerCI = [corr_prePost_one['CI95%'][0][0],
               corr_prePost_both['CI95%'][0][0],
               corr_oneBoth_pre['CI95%'][0][0],
               corr_oneBoth_post['CI95%'][0][0]]
corr_upperCI = [corr_prePost_one['CI95%'][0][1],
               corr_prePost_both['CI95%'][0][1],
               corr_oneBoth_pre['CI95%'][0][1],
               corr_oneBoth_post['CI95%'][0][1]]

#%% create and save a .csv with these results
summaryDict = {'alpha': alphas, 'alpha_upperCI': alpha_upperCIs,
               'alpha_lowerCI': alpha_lowerCIs,
               'corrs': corrs, 'corr_upperCI': corr_upperCI,
               'corr_lowerCI': corr_lowerCI,
               'task': ['one-pleasure precued', 'one-pleasure postcued',
                        'combined-pleasure precued',
                        'combined-pleasure postcued']}
summaryDf = pd.DataFrame(summaryDict)
summaryDf.to_csv(dataDir + '/reliabilities_correlations.csv', index=False)

#%% manual entry of reported alphas and correlations in Brielmann & Pelli (2020)
conditions = ['one precued', 'one postcued', 'both precued', 'both postcued']
alpha_2images = [0.92, 0.89, 0.85, 0.81]
relations = ['one pre vs post', 'both pre vs post', 'pre one vs both',
             'post one vs both']
corrs_2images = [0.93, 0.92, 0.52, 0.45]

#%% plots
# plot all alphas for comparison
fig, ax = plt.subplots()
plt.plot(np.arange(4), alpha_2images, 'ok', label='2-image study')
plt.errorbar(np.arange(4), alphas, yerr=[alpha_posErr, alpha_negErr], fmt='or',
             label='image-music study')
plt.xticks(np.arange(4), conditions)
plt.ylabel('Cronbach\'s ' + r'$\alpha$')
plt.legend(loc='lower right')
plt.show()
plt.close()

# correlations
fig, ax = plt.subplots()
plt.plot(np.arange(4), corrs_2images, 'ok', label='2-image study')
plt.errorbar(np.arange(4), corrs, yerr=[corr_posErr, corr_negErr], fmt='or',
             label='image-music study')
plt.xticks(np.arange(4), relations)
plt.ylabel('Pearsons\'s $r$')
plt.legend(loc='upper right')
plt.show()
plt.close()