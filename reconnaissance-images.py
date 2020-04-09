import glob
from sys import argv
import cv2
import pickle

import shutil
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
import numpy as np
 
# usage: python3 recoimages2018.py k1 category1 category2 baryName verbose
# ATTENTION: les noms de fichiers ne doivent comporter ni - ni espace

#sur ligne de commande: le parametre k de k-means, les 2 répertoires, la racine du nom de fichier de sauvegarde de kmean et regressionlogistique puis un param de verbose
k1 = int(argv[1])

cat1 = argv[2]
cat2 = argv[3]

baryName = argv[4]

if argv[5] == "True":
    verbose = True;
else:
    verbose = False;

listImg=glob.glob(cat1+"/*.jpeg")
tmpa = len(listImg)
listImg += glob.glob(cat2+"/*.jpeg")
               
lesSift = np.empty(shape=(0, 128), dtype=float) # array of all SIFTS from all images
dimImg = [] # nb of sift per file
groundTruth = [0]*tmpa # result we would like to reach
tmpb = len(listImg)-tmpa
groundTruth += [1]*tmpb

for s in listImg:
    if verbose:
        print("###",s,"###")
    image = cv2.imread(s)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp,des = sift.detectAndCompute(gray,None)
    if verbose:
        print("SIFT: ", len(kp))
    dimImg.append(len(des))
    lesSift = np.append(lesSift,des,axis=0)
       
#BOW initialization
bows = np.empty(shape=(0,k1),dtype=float)

# everything ready for the k-means
kmeans1 = KMeans(n_clusters=k1, random_state=0).fit(lesSift)
with open(baryName+'k1.bary', 'wb') as output:
    pickle.dump(kmeans1, output, pickle.HIGHEST_PROTOCOL)

bary1 = kmeans1.cluster_centers_;
if verbose:
    print("result of kmeans 1", kmeans1.labels_)
    
#writing the BOWs for second k-means
i = 0
for nb in dimImg: # for each sound (file)
    tmpBow = [0]*k1
    j = 0
    while j < nb: # for each SIFT of this sound (file)
        tmpBow[kmeans1.labels_[i]] += 1
        j+=1
        i+=1
    copyBow = tmpBow.copy()
    bows = np.append(bows, [copyBow], 0)
#if verbose:
    # print("BOWs : ", bows)

#ready for the logistic regression
logisticRegr = LogisticRegression(max_iter=100)
print(groundTruth)
logisticRegr.fit(bows, groundTruth)
with open(baryName+'.logr', 'wb') as output:
    pickle.dump(logisticRegr, output, pickle.HIGHEST_PROTOCOL)


if verbose:
    res = logisticRegr.predict(bows)
    score = logisticRegr.score(bows, groundTruth)
    print("train score = ", score)
