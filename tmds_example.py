import matplotlib.pyplot as plt
import csv
from TMDS.tmds_implementation import TMDS


totalSize = 100
windowSize = 15
offset = 3
weights = [2,1,1,1,1,3]
indexFloat=[0,1,2,3,4,5]
results =[]
X =[]
colors =[]

#Couleurs des données synthétiques
colorTuple = ('blue','green','black')



filePath = "data/fake_timeGAN_GOOGLE_BIG0.csv"
temp1,temp2 = TMDS(totalSize,windowSize,offset,weights,filePath,-1,indexFloat)
results.append(temp1)
X.append(temp2)
colors.extend([colorTuple[0]]*len(temp1)*len(temp1[0]))

filePath = "data/fake_timeGAN_GOOGLE_BIG1.csv"
temp1,temp2 = TMDS(totalSize,windowSize,offset,weights,filePath,-1, indexFloat)
results.append(temp1)
X.append(temp2)
colors.extend([colorTuple[1]]*len(temp1)*len(temp1[0]))


filePath = "data/GOOGLE_BIG.csv"
temp1,temp2 = TMDS(totalSize,windowSize,offset,weights,filePath,-1, indexFloat)
results.append(temp1)
X.append(temp2)

plt.figure(figsize=(15,8))
plt.scatter(X[:-1],results[:-1],color=colors,s=50,alpha=0.5)

#Les données originales sont en rouge
plt.scatter(X[-1],results[-1],color='red',s=50,alpha=0.5)

print("Drawing...") 
plt.savefig('TMDS/tmds_results')
plt.show()
