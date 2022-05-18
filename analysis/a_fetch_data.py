# -*- coding: utf-8 -*-
"""
Created on September 6, 2021

Update November 30, 2021: for the one case where we have 2 baseline ratings,
                          and to generalize: take average accross baseline 
                          ratings as baseline values
Update April 22, 2022: exclude participants who failed attention check

@author: abrielmann

Basic pre-processing of the data, combining ratings from main task and baseline
Resulting data frame is stored as a .csv file for further use
"""
import os
import numpy as np
import pandas as pd

#%% sett directories
os.chdir('..')
homeDir = os.getcwd()
experiment = 'exp2_sonaLong'
dataDir = homeDir + '/' + experiment

#%% get the raw data
mainDf = pd.read_csv(dataDir + '/all_task.csv')
baseDf = pd.read_csv(dataDir + '/all_baseline.csv')
demoDf = pd.read_csv(dataDir + '/all_questionnaires.csv')

#%% from the raw data, create a new df that contains all info we need for analysis
df = pd.DataFrame({'participant': mainDf['Prolific_ID'],
                   'trialPerBlock': mainDf['Trial.Number'],
                   'runningTrialNumber': mainDf.index,
                   'rating': mainDf['Response'],
                   'rt': mainDf['Reaction.Time'],
                   'cueTime': mainDf['display'],
                   'image': mainDf['ImageStim'],
                   'music': mainDf['MusicStim'],
                   'cued': mainDf['StimCued']})

# set up empty vars for constructing df
baselineImage = []
baselineMusic = []
baselineTarget = []
baselineDistractor = []
cuedImage = []
cuedMusic = []

for trial in df.index:
    peep = df.participant[trial]
    cue = df.cued[trial]
    image = df.image[trial]
    music = df.music[trial]

    imInd = (baseDf['Prolific_ID']==peep) & (baseDf['stimulus']==image)
    baselineImage.append(float(baseDf.loc[imInd, ['Response']].values.mean()))
    musicInd = (baseDf['Prolific_ID']==peep) & (baseDf['stimulus']==music)
    baselineMusic.append(float(baseDf.loc[musicInd, ['Response']].values.mean()))

    if cue == "Image":
        baselineTarget.append(float(baseDf.loc[imInd, ['Response']].values.mean()))
        baselineDistractor.append(float(baseDf.loc[musicInd, ['Response']].values.mean()))
        cuedImage.append(image)
        cuedMusic.append(np.nan)
    elif cue == "Music":
        baselineTarget.append(float(baseDf.loc[musicInd, ['Response']].values.mean()))
        baselineDistractor.append(float(baseDf.loc[imInd, ['Response']].values.mean()))
        cuedImage.append(np.nan)
        cuedMusic.append(music)
    else:
        baselineTarget.append(np.nan)
        baselineDistractor.append(np.nan)
        cuedImage.append(image)
        cuedMusic.append(music)

#%% add the new variables to our df
df['baselineImage'] = baselineImage
df['baselineMusic'] = baselineMusic
df['baselineTarget'] = baselineTarget
df['baselineDistractor'] = baselineDistractor
df['cuedImage'] = cuedImage
df['cuedMusic'] = cuedMusic

#%% create additional new variables
df['baselinesMean'] = df[['baselineImage','baselineMusic']].mean(axis=1)
df['predRating'] = df['baselineTarget'].copy()
df.loc[df['cued']=='Both', 'predRating'] = df.loc[df['cued']=='Both', 'baselinesMean']
df['diffRatePred'] = df['rating'] - df['predRating']

#%% exclude participants
if experiment=='exp1_sonaShort':
    excluded = [4743, 4832, 4787, 4865, 4798]
elif experiment=='exp2_sonaLong':
    excluded = [4828, 4767, 4746, 4855, 4757, 4780, 4882, 4737, 4800]
elif experiment=='exp3_prolificShort':
    excluded = ['5df1fbdb99b2820fcfa3fa37']
else:
    ValueError('Experiment name not known.')

df = df[~df['participant'].isin(excluded)]

#%% save the result as new csv
df.to_csv(dataDir + '/all_data.csv', index=False)