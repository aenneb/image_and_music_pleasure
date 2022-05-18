# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 15:03:27 2021

@author: abrielmann

replicate the leave-one-out model fitting analyses for both-items-cued trials
"""
import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize

#%% SETTINGS
os.chdir('..')
homeDir = os.getcwd()
dataDir = homeDir + '/exp3_prolificShort'

# Do you want to look at pre or postcued?
cue = 'Pre' # 'Pre' or 'Post

#%% define cost functions for the 3 different models
def cost_linear(parameters, data):
    weight = 0.5
    a = parameters[0]
    b = parameters[1]

    imagePleasure = data.baselineImage
    musicPleasure = data.baselineMusic
    ratings = data.rating
    predictions = a+ b*(weight*imagePleasure + (1-weight)*musicPleasure)
    cost = np.sqrt(np.mean((predictions-ratings)**2))
    return cost

def cost_averaging(data):
    predictions = (data.baselineImage + data.baselineMusic)/2
    ratings = data.rating
    cost = np.sqrt(np.mean((predictions-ratings)**2))
    return cost

def cost_linear_modality(parameters, data):
    weight = parameters

    imagePleasure = data.baselineImage
    musicPleasure = data.baselineMusic
    ratings = data.rating
    predictions = weight*imagePleasure + (1-weight)*musicPleasure
    cost = np.sqrt(np.mean((predictions-ratings)**2))
    return cost

#%% settings for optimization
bounds_linear_modality = ((0, 1), ) # bounds for parameter fitting
startValue_linear_modality = [0.5]
bounds_compress = ((0, 10), (0, 1), ) # bounds for parameter fitting
startValue_compress = [0,1]
bounds_extreme = ((-10, 0), (1, 10), ) # bounds for parameter fitting
startValue_extreme = [0, 1]

#%% fetch pre-processed data
df = pd.read_csv(dataDir + '/all_data.csv')
# get some df properties we will need
participants = np.unique(df.participant)
nParticipants = len(participants)
images = np.unique(df.image)
music = np.unique(df.music)
stims = np.concatenate((images, music))
nImages = len(images)
nSongs = len(music)
nStims = nImages+nSongs

#%% Loop through the individual participants
# set up vvariables we want to record
avgRes_linear_modality = []
avgRmse_linear_modality = []
avgRes_compress = []
avgRmse_compress = []
avgRes_extreme = []
avgRmse_extreme = []
avgRmse_averaging = []

for peep in participants:
    peepDf = df.loc[(df['participant']==peep) & (df['cued'] =='Both') & (df['cueTime']==cue)]

    resList_linear_modality = []
    rmseList_linear_modality = []
    resList_compress = []
    rmseList_compress = []
    resList_extreme = []
    rmseList_extreme = []
    rmseList_averaging = []
    
    # We do LOOCV 'by hand', looping through all trials
    for trial in peepDf.index:
        train = peepDf[~peepDf.index.isin([trial])]
        test = peepDf[peepDf.index.isin([trial])]

        res_linear_modality = minimize(cost_linear_modality,
                                       startValue_linear_modality,
                                       args=(train,),
                    bounds=bounds_linear_modality,
                    options={'disp': False, 'maxiter': 1e5, 'ftol': 1e-08})
        resList_linear_modality.append(res_linear_modality.x)
        rmse_linear_modality = cost_linear_modality(res_linear_modality.x,
                                                    test)
        rmseList_linear_modality.append(res_linear_modality.fun)

        res_compress = minimize(cost_linear, startValue_compress, args=(train,),
                    bounds=bounds_compress,
                    options={'disp': False, 'maxiter': 1e5, 'ftol': 1e-08})
        resList_compress.append(res_compress.x)
        rmse_compress = cost_linear(res_compress.x, test)
        rmseList_compress.append(res_compress.fun)

        res_extreme = minimize(cost_linear, startValue_extreme, args=(train,),
                    bounds=bounds_extreme,
                    options={'disp': False, 'maxiter': 1e5, 'ftol': 1e-08})
        resList_extreme.append(res_extreme.x)
        rmse_extreme = cost_linear(res_extreme.x, test)
        rmseList_extreme.append(res_extreme.fun)

        rmse_averaging = cost_averaging(test)
        rmseList_averaging.append(rmse_averaging)

    avgRes_linear_modality.append(np.mean(resList_linear_modality))
    avgRmse_linear_modality.append(np.mean(rmseList_linear_modality))

    avgRes_compress.append(np.mean(resList_compress,axis=0))
    avgRmse_compress.append(np.mean(rmseList_compress))

    avgRes_extreme.append(np.mean(resList_extreme,axis=0))
    avgRmse_extreme.append(np.mean(rmseList_extreme))
    avgRmse_averaging.append(np.mean(rmseList_averaging))

#%% visualize
from matplotlib import pyplot as plt
import seaborn as sns
resDf = pd.DataFrame({'participant': participants,
                    'avgRMSE_averaging': avgRmse_averaging,
                    'avgRMSE_linear_modality': avgRmse_linear_modality,
                    'avgRMSE_compress': avgRmse_compress,
                    'avgRMSE_extreme': avgRmse_extreme,
                    'avgRes_imageWeight_linear': avgRes_linear_modality,
                    'avgRes_a_compress': np.array(avgRes_compress)[:,0],
                    'avgRes_b_compress': np.array(avgRes_compress)[:,1],
                    'avgRes_a_extreme': np.array(avgRes_extreme)[:,0],
                    'avgRes_b_extreme': np.array(avgRes_extreme)[:,1]})
sns.lineplot(data=resDf.iloc[:,1:5])
plt.title(cue + '-cued block')
# Put the legend out of the figure
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xlabel('Participant')
plt.ylabel('Average RMSE')

#%% save these (rough) results
resDf.to_csv(dataDir + '/res_both_' + cue + 'cue.csv', index=False)
