# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 11:21:11 2021

@author: abrielmann

after scoring all questionnaires and running the main analyses, append the
results (especially RMSEs) of LOOCV model fitting to the individual participant
records so we can look at potential relationships between AReA and BMQR and
the model fits/parameters
"""
import os
import pandas as pd

#%% directories; fetch data
os.chdir('..')
homeDir = os.getcwd()
dataDir = homeDir + '/exp3_prolificShort'
demoDf = pd.read_csv(dataDir + '/all_questionnaires.csv')

#%%  get all fit results
resDfPreOne = pd.read_csv(dataDir + '/res_one_Precue.csv')
resDfPreOne.columns.values[1:] = [str(col) + '_preOne' for col in resDfPreOne.columns[1:]]

resDfPostOne = pd.read_csv(dataDir + '/res_one_Postcue.csv')
resDfPostOne.columns.values[1:] = [str(col) + '_postOne' for col in resDfPostOne.columns[1:]]

resDfPreBoth = pd.read_csv(dataDir + '/res_both_Precue.csv')
resDfPreBoth.columns.values[1:] = [str(col) + '_preBoth' for col in resDfPreBoth.columns[1:]]

resDfPostBoth = pd.read_csv(dataDir + '/res_both_Postcue.csv')
resDfPostBoth.columns.values[1:] = [str(col) + '_postBoth' for col in resDfPostBoth.columns[1:]]

# merge (sadly, the safe way only wroks on 2 Dfs at a time)
summaryDf = pd.merge(demoDf, resDfPreOne, on='participant')
summaryDf = pd.merge(summaryDf, resDfPostOne, on='participant')
summaryDf = pd.merge(summaryDf, resDfPreBoth, on='participant')
summaryDf = pd.merge(summaryDf, resDfPostBoth, on='participant')

#%% save the entire thing
summaryDf.to_csv(dataDir + '/results_per_participant.csv', index=False)