# synthetic_data_evaluation

Commandes conda pour installer l'environnement :  

  conda create -n synthetic_data_evaluation python=3.6.11  
  conda activate synthetic_data_evaluation  
  conda install tensorflow=1.10.0  
  conda install scikit-learn  
  conda install theano  
  conda install matplotlib  

Dossier TMDS :  
  Implémentation du TMDS + sortie du scatterplot.  


Dossier timeGAN :  
  Implémentation du timeGAN copiée de https://github.com/jsyoon0823/TimeGAN  
  
  TimeGAN produit des séquences qui n'ont pas de lien entre elles donc il faut créer
  quelques longues séquences au lieu d'un grand nombre de petites séquences.  

  
Dossier data :  
  Données de GOOGLE originales + synthétiques créées par timeGAN. Longueur : 100 entrées.  

  
Dossier cned :  
  cned_data_reader.py permet de transformer les données brutes du cned en des données utilisables par le TMDS.  
  Il faut rajouter un dossier "data" avec les données du cned.

Dossier tSNE :  
  Implémentation du dynamic t-SNE copiée de https://github.com/paulorauber/thesne  

timeGAN_example.py :  
  Permet de générer des séquences de données de la valeur en bourse de Google.  

  
tmds_example.py :  
  Permet de générer une représentation du TMDS de la valeur en bourse de Google.  
  Peut facilement être modifié pour réaliser un TMDS sur les données du cned.  

dynamic_tsne_example.py :  
  Permet de générer une représentation du dynamic t-SNE de la valeur en bourse de Google.
