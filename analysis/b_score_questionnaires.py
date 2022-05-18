# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 10:18:33 2021
Update April 22, 2022: exclude participants who failed attention check

@author: abrielmann

score the questionnaire data from the demographics file and save those
then append it to the raw data
"""
import os
import numpy as np
import pandas as pd

os.chdir('..')
homeDir = os.getcwd()
experiment = 'exp2_sonaLong'
dataDir = homeDir + '/' + experiment

#%%
demoDf = pd.read_csv(dataDir + '/all_questionnaires.csv')

#%% AReA can be scored as both sum- or mean; let's do sums for greater variance
#  Aesthetic Appreciation <-- items 1, 2, 3, 4, 6, 9, 13, 14
#  Intense Aesthetic Experience <-- items 8, 11, 12, 13
#  Creative Behavior <-- items 5, 7, 10
aesthAppIdxs = [1, 2, 3, 4, 6, 9, 13, 14]
intAesthExpIdxs = [8, 11, 12, 13]
creatBehavIdxs = [5, 7, 10]

demoDf['AReA_aesthApp'] = demoDf.iloc[:, aesthAppIdxs].values.sum(axis=1)
demoDf['AReA_intAesthExp'] = demoDf.iloc[:, intAesthExpIdxs].values.sum(axis=1)
demoDf['AReA_creatBehav'] = demoDf.iloc[:, creatBehavIdxs].values.sum(axis=1)
demoDf['AReA_total'] = demoDf.iloc[:, np.arange(1,15)].values.sum(axis=1)

#%% BMRQ - a total sum score and factor-weight based subscale scoring
demoDf['BMRQ_sum'] = demoDf.iloc[:, np.arange(15,35)].values.sum(axis=1)

# BMRQ subscale scores need to be calculated carefully
# see js implementation view-source:http://brainvitge.org/z_oldsite/bmrq.php
AMS = []
AEE = []
AMR = []
ASM = []
ASC = []
ARW = []

for peep in demoDf.participant:
    thisDf = demoDf.loc[demoDf.participant==peep]
    items = thisDf.iloc[:,np.arange(15,35)].values.flatten()
    a1=(items[0]-3.85180)/0.90570
    a2=(items[1]-1.72700)/1.06350
    a3=(items[2]-4.34420)/0.83540
    a4=(items[3]-4.17040)/0.94910
    a5=(items[4]-1.64880)/1.05530
    a6=(items[5]-3.74210)/0.89520
    a7=(items[6]-3.88800)/0.97830
    a8=(items[7]-4.53210)/0.68650
    a9=(items[8]-4.26140)/0.81850
    a10=(items[9]-3.95800)/1.06150
    a11=(items[10]-3.45510)/1.08970
    a12=(items[11]-3.55190)/1.27980
    a13=(items[12]-3.27650)/1.30490
    a14=(items[13]-4.34660)/0.80310
    a15=(items[14]-4.28940)/0.99190
    a16=(items[15]-3.81680)/0.94980
    a17=(items[16]-2.28590)/1.13310
    a18=(items[17]-3.94400)/0.97470
    a19=(items[18]-4.11550)/0.86440
    a20=(items[19]-3.99650)/0.99350

    MS=-(0.0055*a1)-(0.4190*a2)+(0.0376*a3)+(0.1086*a4)+(0.0038*a5)+(0.0460*a6)+(0.2751*a7)-(0.0151*a8)-(0.0607*a9)-(0.0401*a10)+(0.4148*a11)-(0.0250*a12)-(0.0423*a13)+(0.0159*a14)+(0.0937*a15)+(0.0536*a16)+(0.2106*a17)+(0.0083*a18)-(0.0603*a19)+(0.0025*a20)
    EE=(0.0236*a1)-(0.0099*a2)+(0.2998*a3)+(0.0198*a4)-(0.0517*a5)-(0.0033*a6)+(0.0105*a7)+(0.2989*a8)+(0.0222*a9)-(0.0242*a10)-(0.0492*a11)+(0.3307*a12)+(0.0006*a13)-(0.0254*a14)+(0.0187*a15)-(0.0175*a16)+(0.0601*a17)+(0.3598*a18)+(0.0271*a19)-(0.0077*a20)
    MR=(0.0336*a1)-(0.0630*a2)+(0.0536*a3)+(0.2432*a4)+(0.1130*a5)-(0.0486*a6)+(0.0122*a7)+(0.0620*a8)+(0.3327*a9)-(0.0282*a10)-(0.0336*a11)-(0.1296*a12)-(0.0248*a13)+(0.3776*a14)+(0.1671*a15)+(0.0534*a16)-(0.1572*a17)-(0.0384*a18)+(0.2663*a19)+(0.0290*a20)
    SM=-(0.0602*a1)+(0.0557*a2)-(0.0036*a3)+(0.0163*a4)-(0.3377*a5)+(0.0494*a6)+(0.0258*a7)-(0.0127*a8)+(0.0107*a9)+(0.4263*a10)+(0.0645*a11)+(0.0838*a12)-(0.0048*a13)+(0.0110*a14)+(0.1481*a15)+(0.0266*a16)-(0.0631*a17)-(0.0440*a18)-(0.0042*a19)+(0.3077*a20)
    SC=(0.3566*a1)-(0.0714*a2)-(0.1191*a3)-(0.0804*a4)-(0.1072*a5)+(0.3277*a6)-(0.1801*a7)-(0.0092*a8)-(0.0121*a9)-(0.0818*a10)-(0.0978*a11)+(0.0338*a12)+(0.4228*a13)-(0.0646*a14)-(0.1048*a15)+(0.2254*a16)+(0.2533*a17)+(0.0281*a18)+(0.1319*a19)-(0.0015*a20)
    RW=(0.0782*a1)-(0.1236*a2)+(0.0978*a3)+(0.102*a4)-(0.1021*a5)+(0.0821*a6)+(0.0489*a7)+(0.1089*a8)+(0.1039*a9)+(0.0856*a10)+(0.0755*a11)+(0.0913*a12)+(0.0734*a13)+(0.1108*a14)+(0.1061*a15)+(0.0841*a16)+(0.0575*a17)+(0.0955*a18)+(0.1104*a19)+(0.1013*a20)

    AMS.append(np.round(MS*10+50))
    AEE.append(np.round(EE*10+50))
    AMR.append(np.round(MR*10+50))
    ASM.append(np.round(SM*10+50))
    ASC.append(np.round(SC*10+50))
    ARW.append(np.round(RW*10+50))

demoDf['BMRQ_musicSeeking'] = AMS
demoDf['BMRQ_emoEvoc'] = AEE
demoDf['BMRQ_moodReg'] = AMR
demoDf['BMRQ_sensMot'] = ASM
demoDf['BMRQ_social'] = ASC
demoDf['BMRQ_musicRew'] = ARW

#%% exclude participants
if experiment=='exp1_sonaShort':
    excluded = [4743, 4832, 4787, 4865, 4798]
elif experiment=='exp2_sonaLong':
    excluded = [4828, 4767, 4746, 4855, 4757, 4780, 4882, 4737, 4800]
elif experiment=='exp3_prolificShort':
    excluded = ['5df1fbdb99b2820fcfa3fa37 ']
else:
    ValueError('Experiment name not known.')

demoDf = demoDf[~demoDf['participant'].isin(excluded)]

#%% Save the new df including scores
demoDf.to_csv(dataDir + '/all_questionnaires.csv', index=False)