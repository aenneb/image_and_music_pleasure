# -*- coding: utf-8 -*-
"""
Created on Wed Dec 8, 2021

@author: abrielmann

 look at potential relationships between AReA and BMQR and
the model fits/parameters
"""
import os
import pandas as pd
import scipy.stats

#%% directories
os.chdir('..')
homeDir = os.getcwd()
dataDir = homeDir + '/exp1_sonaShort'

#%% fetch dattta
df = pd.read_csv(dataDir + '/results_per_participant.csv')
firstResCol = df.columns.get_loc('avgRMSE_accurate_preOne')
firstScoreCol = df.columns.get_loc('AReA_aesthApp')

#%% Let's do one correlation matrix for all scale scores
scoreDf = df.iloc[:, firstScoreCol:firstScoreCol+11]
resDf = df.iloc[:, firstResCol:]

scoreNames = scoreDf.columns
resNames = resDf.columns

#%% we loop through all scores and all results and correlate
scoreList = []
resList = []
rhoList = []
pList = []
for score in scoreNames:
    for res in resNames:
        rho, pval = scipy.stats.spearmanr(df[score], df[res])
        scoreList.append(score)
        resList.append(res)
        rhoList.append(rho)
        pList.append(pval)

# merge those results into a df for easier handling, saving
corrDict = {'score': scoreList, 'res': resList, 'rho': rhoList, 'p': pList}
corrDf = pd.DataFrame(corrDict)

#%% display correlations that are, by standard cut-offs, significant
print(corrDf[corrDf.p<.05])

#%% the above is a long table for exp3 --> pack it into a df, save as csv
sigCorr = corrDf[corrDf.p<.05]
sigCorr.to_csv(dataDir + '/significantCorrelations_questionnaires_fits.csv')