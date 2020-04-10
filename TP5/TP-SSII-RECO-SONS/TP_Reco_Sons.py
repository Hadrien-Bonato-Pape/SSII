import glob
from sys import argv
from python_speech_features import mfcc
import librosa
#import shutil
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
# usage: python3 recosons.py k1 k2 verbose
# ATTENTION: les noms de fichiers ne doivent comporter ni - ni espace

#sur ligne de commande: les 2 parametres de k means puis un param de verbose
k1 = 2
k2 = 2


if "True" == "True":
    verbose = True;
else:
    verbose = False;

hadrien=glob.glob("/Users/hbp/Desktop/TP-SSII-RECO-SONS/archive/Hadrien/*.wav")
kevin=glob.glob("/Users/hbp/Desktop/TP-SSII-RECO-SONS/archive/Kevin/*.wav")

lesMfccH = np.empty(shape=(0, 13), dtype=float) # array of all MFCC from all sounds
dimSonsH = [] # nb of mfcc per file


lesMfccK = np.empty(shape=(0, 13), dtype=float) # array of all MFCC from all sounds
dimSonsK = [] # nb of mfcc per file

lesMfcc = np.empty(shape=(0, 13), dtype=float) # array of all MFCC from all sounds
dimSons = [] # nb of mfcc per file

for s in hadrien:
    if verbose:
        print("###",s,"###")
        
    (sig,rate) = librosa.load(s)
    mfcc_feat = mfcc(sig,rate,nfft=1024)
    
    if verbose:
        print("MFCC: ", mfcc_feat.shape)
        
    dimSons.append(mfcc_feat.shape[0])
    lesMfcc = np.append(lesMfcc, mfcc_feat, axis=0)
    lesMfccH = np.append(lesMfccH, mfcc_feat, axis=0)


for s in kevin:
    if verbose:
        print("###",s,"###")
        
    (sig,rate) = librosa.load(s)
    mfcc_feat = mfcc(sig,rate, nfft=1024)
    
    if verbose:
        print("MFCC: ", mfcc_feat.shape)
        
    dimSons.append(mfcc_feat.shape[0])
    lesMfcc = np.append(lesMfcc, mfcc_feat, axis=0)
    lesMfccK = np.append(lesMfccK, mfcc_feat, axis=0)

       
#BOW initialization
bows = np.empty(shape=(0,k1),dtype=int)

# everything ready for the 1st k-means
kmeans1 = KMeans(n_clusters=k1, random_state=0).fit(lesMfcc)

if verbose:
    print("result of kmeans 1", kmeans1.labels_)

plt.scatter(lesMfccH[:, 0], lesMfccH[:, 1], c='blue', s=50, cmap='viridis')
plt.scatter(lesMfccK[:, 0], lesMfccK[:, 1], c='red', s=50, cmap='viridis')

centers = kmeans1.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
plt.show()




#writing the BOWs for second k-means
i = 0
for nb in dimSons: # for each sound (file)
    tmpBow = np.array([0]*k1)
    j = 0
    while j < nb: # for each MFCC of this sound (file)
        tmpBow[kmeans1.labels_[i]] += 1
        j+=1
        i+=1
    tmpBow = tmpBow / nb
    copyBow = tmpBow.copy()
    bows = np.append(bows, [copyBow], 0)
if verbose:
    print("nb of MFCC vectors per file : ", dimSons)
    print("BOWs : ", bows)

#ready for second k-means
kmeans2 = KMeans(n_clusters=k2, random_state=0).fit(bows)
if verbose:
    print("result of kmeans 2", kmeans2.labels_)
    

hadrienVoice = "Hadrien"
kevinVoice = "Kevin"

listSons = glob.glob(hadrienVoice + "/*.wav")
tmpa = 10 #on mémorise le nb d'éléments de la première classe
listImg = glob.glob(kevinVoice + "/*.wav")

#liste des labels:
groundTruth = [0]*tmpa
tmpb = 10 #nb. éléments de la snde classe
groundTruth += [1]*tmpb

### [compute MFCCs, BOWs, ...as before] and:
#ready for the logistic regression
logisticRegr = LogisticRegression()
logisticRegr.fit(bows, groundTruth)

#affichage des scores
res = logisticRegr.predict(bows)
score = logisticRegr.score(bows, groundTruth)
print("train score = ", score)



##lesMfcc = np.empty(shape=(0, 13), dtype=float) # array of all MFCC from all sounds
##dimSons = [] # nb of mfcc per file
##(sig,rate) = librosa.load("sons-mystere/SalutMystere-H.wav")
##mfcc_feat = mfcc(sig,rate,nfft=1024)
##dimSons.append(mfcc_feat.shape[0])
##lesMfcc = np.append(lesMfcc,mfcc_feat,axis=0)
##
### everything ready for the 1st k-means
##prediction = kmeans2.predict(lesMfcc)
##print("Prediciton : ", prediction)







