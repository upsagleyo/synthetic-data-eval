import numpy as np
import os
from sklearn.utils import check_random_state
from sklearn.preprocessing import StandardScaler

from tSNE.model.dynamic_tsne import dynamic_tsne
from TMDS.tmds_implementation import firstDistanceMatrix, distanceMatrix
import csv

import matplotlib.pyplot as plt

def distanceMatrixArray(superListEntries,weights):
    weights = weights/np.sum(weights)
    MList = []
    
    MList.append(firstDistanceMatrix(superListEntries[0],weights))
    
    for i in range(1,len(superListEntries)):
        MList.append(distanceMatrix(MList[-1],superListEntries[i],1,weights))
        
    return MList


def compute_and_print_dynamic_tSNE(realData,fakeData):
    
    seed = 0
    print("Computing real data dynamic t-SNE...")
    Yreal = dynamic_tsne(realData,verbose=True,lmbda=0.5,n_epochs=400,perplexity=5,
                         random_state=seed,initial_lr=10,final_lr=2,metric='precomputed')
    print("Computing fake data dynamic t-SNE...")
    Yfake = dynamic_tsne(fakeData,verbose=True,lmbda=0.5,n_epochs=400,perplexity=5,
                         random_state=seed,initial_lr=10,final_lr=2,metric='precomputed')
    timelen = len(Yreal)
    for t in range(timelen):
        plt.figure()
        plt.scatter(Yreal[t][:, 0], Yreal[t][:, 1],color = 'red')
        plt.scatter(Yfake[t][:, 0], Yfake[t][:, 1],color = 'blue',alpha=0.5)
        plt.savefig('tSNE/dynamic_results/'+ str(t))
        plt.show()
        plt.close()

        

def data_loading(totalSize, windowSize, offset, filePath, indexTimestamp,indexFloat,header):
    #totalSize : nombre total d'entrées à considérer
    #windowSize : taille de la  fenêtre glissante
    #offset : taille du décalage entre chaque fenêtres
    #weights : liste des poids de chaque paramètre/variable aléatoire. Liste de taille n, n étant le nombre de paramètres
    #filePath : chemin du fichier csv
    #indexTimestamp : index du timestamp dans les entrées, 
    #mettre -1 s'il n'y en a pas et que les entrées sont par ordre chronologique
    #indexFloat : liste des indexes des données à valeurs réelles
    #header : est-ce que le tableau dispose d'une ligne avec des titres ou non.
    
    #ouverture du fichier
    file = open(filePath, "r")
    rawData = list(csv.reader(file, delimiter=","))[header:totalSize+header]
    file.close()
    
    #Conversion des timestamps de string à float
    if indexTimestamp!=-1:
            for record in rawData :
                record[indexTimestamp] = float(record[indexTimestamp])
            rawData.sort(key=lambda x: x[indexTimestamp])
    
    #Conversion en float et normalisation des données numériques (timestamp inclus)
    for index in indexFloat :
        maximum = max([float(i) for i in rawData[:][index]])
        minimum = min([float(i) for i in rawData[:][index]])
        for record in rawData:
            record[index] = (float(record[index])-minimum)/(maximum-minimum)
            
    superListEntries = []
    nbSteps = int((totalSize-windowSize)/offset)
    
    for step in range(nbSteps):
        t=offset*step
        superListEntries.append(rawData[t:t+windowSize])
    
    return superListEntries








weights = [1,1,1,1,1,1]
totalSize = 100
windowSize = 15
offset = 3
indexFloat = [0,1,2,3,4,5]
indexTimestamp =-1

filePath = "data/fake_timeGAN_GOOGLE_BIG0.csv"
superListEntries =data_loading(totalSize, windowSize, offset, filePath, indexTimestamp,indexFloat,False)
realData = distanceMatrixArray(superListEntries,weights)

filePath = "data/GOOGLE_BIG.csv"
superListEntries =data_loading(totalSize, windowSize, offset, filePath, indexTimestamp,indexFloat,True)
fakeData= distanceMatrixArray(superListEntries,weights)

compute_and_print_dynamic_tSNE(realData,fakeData)
