# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22, 2022

@author: abrielmann

"""
import os
import pandas as pd

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

#%%
print(rmseDf.groupby(['cue', 'model', 'task']).mean()['avgRMSE'])