import csv
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.manifold import MDS

#Distance entre deux entrées catégorielles
def categoricalDistance (A,B,weights):
    dim = len(A)
    
    if dim!=len(B):
        raise Exception("Arrays of different sizes")
       
    for i in range(dim):
        if type(A[i])==float:
            sum+=((A[i]-B[i])**2)*weights[i]
        else :
            sum+=int(A[i]!=B[i])*weights[i]
    return np.sqrt(sum)
#Calcul de la première matrice
def firstDistanceMatrix(listEntries,weights):
    
    length = len(listEntries)
    
    M = np.zeros((length,length))
    
    for i in range(length) :
        for j in range(i+1,length) :
            dist = categoricalDistance(listEntries[i],listEntries[j],weights)
            M[i][j] = dist
            M[j][i] = dist
            
    return M

#Calcul des matrices suivantes
def distanceMatrix(oldM,listEntries,stepSize,weights):
    #oldM : matrice de distance à t-1
    #listEntries : liste des entrées : dimension = nb_entrées x taille entrée 
    length = len(oldM)
    
    M = np.zeros((length,length))
        
    for i in range(length-stepSize):
        for j in range(length-stepSize):
            M[i][j]=oldM[i+stepSize,j+stepSize]
    #Upper left corner
    
    for i in range(length-stepSize, length):
        for j in range(length):
            dist = categoricalDistance(listEntries[i],listEntries[j],weights)
            M[i][j]= dist
            M[j][i]= dist
            
    return M

#Slice Flipping : vérifie si une majorité d'entrées ont changé de signe depuis la fenêtre précédente
def flipSlice(results,old_results,offset):
    count = 0
    for i in range(len(results)-offset):
        if(results[i+offset][0]*old_results[i][0]<0):
            count+=1
    if count > len(results)/2 :
        results = -results
    return results

def TMDS(totalSize, windowSize, offset, weights, filePath, indexTimestamp,indexFloat):
    #totalSize : nombre total d'entrées à considérer
    #windowSize : taille de la  fenêtre glissante
    #offset : taille du décalage entre chaque fenêtres
    #weights : liste des poids de chaque paramètre/variable aléatoire. Liste de taille n, n étant le nombre de paramètres
    #filePath : chemin du fichier csv
    #indexTimestamp : index du timestamp dans les entrées, 
    #mettre -1 s'il n'y en a pas et que les entrées sont par ordre chronologique
    #indexFloat : liste des indexs des données à valeurs réelles
    
    mds = MDS(random_state=0,n_components = 1,n_jobs=-1,dissimilarity = 'precomputed')
    
    weights = weights/np.sum(weights)
    nbSteps = int((totalSize-windowSize)/offset)
    
    #Ouverture du fichier csv
    file = open(filePath, "r")
    realData = list(csv.reader(file, delimiter=","))[1:totalSize+1]
    file.close()
    
    #Conversion des timestamps de string à float
    if indexTimestamp!=-1:
        for record in realData :
            record[indexTimestamp] = float(record[indexTimestamp])
        realData.sort(key=lambda x: x[indexTimestamp])

    #Conversion en float et normalisation des données numériques (timestamp inclus)
    for index in indexFloat :
        maximum = max([float(i) for i in realData[:][index]])
        minimum = min([float(i) for i in realData[:][index]])
        for record in realData :
            record[index] = (float(record[index])-minimum)/(maximum-minimum)
        
   
    
    #Calcul de la matrice des distances pour la première fenêtre + MDS
    XList=[]
    resultList=[]
    t=0
    
    
    listEntries = realData[0:windowSize][:len(weights)]
    matrix = firstDistanceMatrix(listEntries,weights)
    results=mds.fit_transform(matrix)
    X = [0 for i in range(len(results))]
    
    XList.append(X)
    resultList.append(results)
    
   
    print(t+1,'/',nbSteps)    
    
    #Calcul des matrices suivantes
    for step in range(1,nbSteps):
        t = offset*step
        listEntries = realData[t:t+windowSize]
        
        #Calcul de la nouvelle matrice
        matrix = distanceMatrix(matrix,listEntries,offset,weights)
        
        #Slice flipping + MDS
        results = flipSlice(mds.fit_transform(matrix),resultList[-1],offset)
        
    
    
        #Affichage de la progression du programme
        if (step%(math.ceil(nbSteps/20)) ==0 or step+1==nbSteps):
            print(step+1,'/',nbSteps)
    
        #Abscisse des points (artéfact de l'ancien code, très optimisable)
        X = [step for i in range(len(results))]
        
        XList.append(X)
        resultList.append(results)
    
    
    
    return resultList,XList
