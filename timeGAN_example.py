# %load main.py
'''
2019 NeurIPS Submission
Title: Time-series Generative Adversarial Networks
Authors: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar

Last Updated Date: May 29th 2019
Code Author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

Main Function
- Import Dataset
- Generate Synthetic Dataset
- Evaluate the performances in three ways
(1) Visualization (t-SNE, PCA)
(2) Discriminative Score
(3) Predictive Score

Inputs
- Dataset
- Network Parameters

Outputs
- time-series synthetic data
- Performances
(1) Visualization (t-SNE, PCA)
(2) Discriminative Score
(3) Predictive Score
'''

#%% Necessary Packages

import numpy as np
import sys

#%% Functions
# 1. Models
from timeGAN.tgan import tgan

# 2. Data Loading
from timeGAN.data_loading import google_data_loading, sine_data_generation

# 3. Metrics
sys.path.append('timeGAN/metrics')
from discriminative_score_metrics import discriminative_score_metrics
from visualization_metrics import PCA_Analysis, tSNE_Analysis
from predictive_score_metrics import predictive_score_metrics

#%% Main Parameters
# Data
data_set = ['google','sine']
data_name = data_set[0]

# Experiments iterations
Iteration = 1
Sub_Iteration = 10

#%% Data Loading
seq_length = 100
data_length = 150

if data_name == 'google':
    dataX,denominator = google_data_loading(seq_length,data_length)
elif data_name == 'sine':
    No = 10000
    F_No = 5
    dataX = sine_data_generation(No, seq_length, F_No)

print(data_name + ' dataset is ready.')

#%% Newtork Parameters
parameters = dict()
print(np.shape(dataX))
parameters['hidden_dim'] = len(dataX[0][0,:])*4
parameters['num_layers'] = 3
parameters['iterations'] = 1000
parameters['batch_size'] = 12
parameters['module_name'] = 'gru'   # Other options: 'lstm' or 'lstmLN'
parameters['z_dim'] = len(dataX[0][0,:]) 

#%% Experiments
# Output Initialization
Discriminative_Score = list()
Predictive_Score = list()

# Each Iteration
for it in range(Iteration):
    
    # Synthetic Data Generation
    dataX_hat = tgan(dataX, parameters)   
      
    print('Finish Synthetic Data Generation')
    
    
    #Ecriture des résultats
    dataX_hat2 = dataX_hat
    dataX_hat2 = np.flip(dataX_hat2,axis=1)*(denominator+ 1e-7)+49.274517

    np.savetxt("data/fake_timeGAN_GOOGLE_BIG0.csv", dataX_hat2[0], delimiter=",",fmt='%f')
    np.savetxt("data/fake_timeGAN_GOOGLE_BIG1.csv", dataX_hat2[1], delimiter=",",fmt='%f')
    np.savetxt("data/fake_timeGAN_GOOGLE_BIG2.csv", dataX_hat2[2], delimiter=",",fmt='%f')
    np.savetxt("data/fake_timeGAN_GOOGLE_BIG3.csv", dataX_hat2[3], delimiter=",",fmt='%f')
    np.savetxt("data/fake_timeGAN_GOOGLE_BIG4.csv", dataX_hat2[4], delimiter=",",fmt='%f')

    fullData = []
    for i in range(len(dataX_hat2)):
        fullData.extend(dataX_hat2[i])

    np.savetxt("data/fake_timeGAN_GOOGLE_BIG_FULL.csv",fullData,delimiter=",",fmt='%f')

    #%% Performance Metrics
    
    # 1. Discriminative Score
    Acc = list()
    for tt in range(Sub_Iteration):
        Temp_Disc = discriminative_score_metrics (dataX, dataX_hat)
        Acc.append(Temp_Disc)
    
    Discriminative_Score.append(np.mean(Acc))
    
    # 2. Predictive Performance
    MAE_All = list()
    for tt in range(Sub_Iteration):
        MAE_All.append(predictive_score_metrics (dataX, dataX_hat))
        
    Predictive_Score.append(np.mean(MAE_All))        


#%% 3. Visualization
PCA_Analysis (dataX, dataX_hat)
tSNE_Analysis (dataX, dataX_hat)

# Print Results
print('Discriminative Score - Mean: ' + str(np.round(np.mean(Discriminative_Score),4)) + ', Std: ' + str(np.round(np.std(Discriminative_Score),4)))
print('Predictive Score - Mean: ' + str(np.round(np.mean(Predictive_Score),4)) + ', Std: ' + str(np.round(np.std(Predictive_Score),4)))






