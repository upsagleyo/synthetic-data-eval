'''
2019 NeurIPS Submission
Title: Time-series Generative Adversarial Networks
Authors: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar

Last Updated Date: May 29th 2019
Code Author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

Data loading
(1) Load Google dataset
- Transform the raw data to preprocessed data
(2) Generate Sine dataset

Inputs
(1) Google dataset
- Raw data
- seq_length: Sequence Length
(2) Sine dataset
- No: Sample Number
- T_No: Sequence Length
- F_No: Feature Number

Outputs
- time-series preprocessed data
'''

#%% Necessary Packages
import numpy as np
import os

#%% Min Max Normalizer

def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    #print(numerator / (denominator + 1e-7))
    return numerator / (denominator + 1e-7),denominator

#%% Load Google Data
    
def google_data_loading (seq_length,data_length):

    # Load Google Data
    x = np.loadtxt(os.path.join(os.path.dirname(__file__), '..', 'data', 'GOOGLE_BIG.csv'), delimiter = ",",skiprows = 1)[:data_length]
    # Flip the data to make chronological data
    x = x[::-1]
    # Min-Max Normalizer
    x,denominator= MinMaxScaler(x)
    
    # Build dataset
    dataX = []
    print(np.shape(x))
    # Cut data by sequence length
    for i in range(0, len(x) - seq_length+1):
        _x = x[i:i + seq_length]
        dataX.append(_x)
    # Mix Data (to make it similar to i.i.d)
    
    idx = np.random.permutation(len(dataX))
    
    outputX = []
    for i in range(len(dataX)):
        outputX.append(dataX[idx[i]])
        
        
        
        
    
    return outputX,denominator
  
#%% Sine Data Generation

def sine_data_generation (No, T_No, F_No):
  
    # Initialize the output
    dataX = list()

    # Generate sine data
    for i in range(No):
      
        # Initialize each time-series
        Temp = list()

        # For each feature
        for k in range(F_No):              
                          
            # Randomly drawn frequence and phase
            freq1 = np.random.uniform(0,0.1)            
            phase1 = np.random.uniform(0,0.1)
          
            # Generate Sine Signal
            Temp1 = [np.sin(freq1 * j + phase1) for j in range(T_No)] 
            Temp.append(Temp1)
        
        # Align row/column
        Temp = np.transpose(np.asarray(Temp))
        
        # Normalize to [0,1]
        Temp = (Temp + 1)*0.5
        
        dataX.append(Temp)
                
    return dataX
    
