import json
import os
import re
import time
import csv
import datetime
import copy
import numpy as np


def dataTable(fullData):
    listKeys = list(fullData[0].keys())
    outputTable = [listKeys]
    
    for dictionaries in fullData :
        outputTable.append(list(dictionaries.values()))
        
    return outputTable

def readJson(path):

    listFiles =os.listdir(path)

    fullData = []
    notes = dict(list())
    
    for fileName in listFiles :
        filePath = path+"/"+fileName
        f = open(filePath)
        data = json.load(f) #liste de dictionnaires
        
        #On passe une première fois pour les actions viewed, enlever les 'submit' et changer les verbes
        
        i=0
        
        while i < len(data) :
            
            indexDict = data[i]
            indexDict['verb'] = indexDict['verb']['display']['en']

            
            if indexDict['verb']=='submit' :
                del data[i]
                
            if indexDict['verb']=='viewed' :
                del indexDict['version']
                
                indexDict['actor'] = re.search("[0-9]+",indexDict['actor']['mbox']).group()

                indexDict.update({'ressourceID':0})

                objectString = indexDict['object']['id']
                objectString = re.search("[0-9]+" , objectString)
                if objectString.group() != "1562":
                    indexDict['ressourceID'] = objectString.group()
                del indexDict['object']

                if 'context' in indexDict:
                    del indexDict['context']
                try:
                    indexDict['timestamp'] = time.mktime(datetime.datetime.strptime(indexDict['timestamp'][:-6],
                                                     "%Y-%m-%dT%H:%M:%S").timetuple())
                except ValueError:
                     indexDict['timestamp'] = time.mktime(datetime.datetime.strptime(indexDict['timestamp'][:-6],
                                                     "%Y-%m-%dT%H").timetuple())
                
                i+=1
            
            if indexDict['verb']=='scored' :
    
                indexDict['actor'] = re.search("[0-9]+",indexDict['actor']['mbox']).group()
                actor = indexDict['actor']
                
                minimum = indexDict['result']['score']['min']
                maximum = indexDict['result']['score']['max']
                raw = indexDict['result']['score']['raw']
                del data[i]
                
                score = (raw-minimum)/(maximum-minimum)
                if actor in notes :
                    notes[actor].append(score)
                else :
                    notes.update({actor : list()})
                    notes[actor].append(score)
        
        fullData.extend(data)
        f.close()
    
    
    #Ajout des moyennes
    for indexDict2 in fullData :
        actor = str(indexDict2['actor'])
            
        if actor in notes :
            indexDict2.update({'score' : np.mean(notes[actor])})
                
        else :
            indexDict2.update({'score' : np.nan})
                
                
                
    print('Done')
    outputTable = dataTable(fullData)   
    return outputTable


def writeJson(path="data"):
    myjson = readJson(path)

    header = myjson[0]
    data = myjson[1:]
    with open('cned_edited.csv', 'w', encoding='UTF8', newline='') as f:

        writer = csv.writer(f)

        # write the header
        writer.writerow(header)

        # write multiple rows
        writer.writerows(data)
        
        
        

writeJson()