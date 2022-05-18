# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 2022

@author: abrielmann

check whether ratings systematically change over time
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt
import seaborn as sns

os.chdir('..')
os.chdir('..')
homeDir = os.getcwd()
experiment = 'exp3_prolificShort'
dataDir = homeDir + '/' + experiment

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

#%% corr w trial number per participant
corr_per_peep = []
corr_sig_per_peep = []
for peep in participants:
    thisDf = df[df.participant==peep]
    r, p = stats.pearsonr(thisDf.runningTrialNumber, thisDf.rating)
    corr_per_peep.append(r)
    corr_sig_per_peep.append(p)
    
num_sig_peep = np.sum([r<0.05 for r in corr_sig_per_peep])
print(str(num_sig_peep) + ' sig corrs per participant')
print(str(np.median(corr_per_peep)) + ' median corr per participant')
print(stats.wilcoxon(corr_per_peep))

plt.hist(corr_per_peep)
plt.vlines(0, 0, 7, 'k', 'dashed', linewidth=2)
plt.vlines(np.median(corr_per_peep), 0, 7, 'r', linewidth=2)
plt.ylabel('# participants')
plt.xlabel('$r$')
sns.despine()
plt.show()
plt.close()

#%% corr w trial number per image
corr_per_im = []
corr_sig_per_im = []
for im in images:
    thisDf = df[df.image==im]
    r, p = stats.pearsonr(thisDf.runningTrialNumber, thisDf.rating)
    corr_per_im.append(r)
    corr_sig_per_im.append(p)
    
num_sig_im = np.sum([r<0.05 for r in corr_sig_per_im])
print(str(num_sig_im) + ' sig corrs per image')
print(str(np.median(corr_per_im)) + ' median corr per image')
print(stats.wilcoxon(corr_per_im))

plt.hist(corr_per_im)
plt.vlines(0, 0, 6, 'k')
plt.ylabel('# images')
plt.xlabel('$r$')
sns.despine()
plt.show()
plt.close()

#%% corr w trial number per song
corr_per_song = []
corr_sig_per_song = []
for song in music:
    thisDf = df[df.music==song]
    r, p = stats.pearsonr(thisDf.runningTrialNumber, thisDf.rating)
    corr_per_song.append(r)
    corr_sig_per_song.append(p)
    
num_sig_song = np.sum([r<0.05 for r in corr_sig_per_song])
print(str(num_sig_song) + ' sig corrs per song')
print(str(np.median(corr_per_song)) + ' median corr per song')
print(stats.wilcoxon(corr_per_song))

plt.hist(corr_per_song)
plt.vlines(0, 0, 6, 'k')
plt.ylabel('# songs')
plt.xlabel('$r$')
sns.despine()
plt.show()
plt.close()

#%% summary boxplots
plt.violinplot([corr_per_peep, corr_per_im, corr_per_song])
plt.hlines(0, 0.5, 3.5, 'k', 'dashed', linewidth=2)
plt.xticks(np.arange(1,4),['participant', 'image', 'song'])
plt.ylabel('$r$')
sns.despine()
plt.show()
plt.close()
