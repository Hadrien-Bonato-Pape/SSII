import glob
from sys import argv
import cv2
import pickle

import shutil
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import f1_score

k1 = int(argv[1])

cat1 = argv[2]
cat2 = argv[3]

baryName = argv[4]

listImg=glob.glob(cat1+"/*.jpeg")
tmpa = len(listImg)
listImg += glob.glob(cat2+"/*.jpeg")

print("Catégorie " + cat1 + "est normalement 0.")
print("Catégorie " + cat2 + "est normalement 1.")
               
lesSift = np.empty(shape=(0, 128), dtype=float) # array of all SIFTS from all images
dimImg = [] # nb of sift per file
groundTruth = [0]*tmpa # result we would like to reach
tmpb = len(listImg)-tmpa
groundTruth += [1]*tmpb

for s in listImg:
    print("###",s,"###")
    image = cv2.imread(s)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp,des = sift.detectAndCompute(gray,None)
    print("SIFT: ", len(kp))
    dimImg.append(len(des))
    lesSift = np.append(lesSift,des,axis=0)
       
with open(baryName+'k.bary',  'rb') as input:
    km1 = pickle.load(input)

bows = np.empty(shape=(0,k1),dtype=float)
km1.predict(lesSift)

i = 0
for nb in dimImg:
    tmpBow = [0]*k1
    j = 0
    while j < nb:
        tmpBow[km1.labels_[i]] += 1
        j+=1
        i+=1
    copyBow = tmpBow.copy()
    bows = np.append(bows, [copyBow], 0)

with open(baryName+'L.logr',  'rb') as input:
    logisticRegr = pickle.load(input)


res = logisticRegr.predict(bows)

print(res)
print(groundTruth)

score = logisticRegr.score(bows, groundTruth)
print("f1 score = ",f1_score(groundTruth, res, average='binary'))
print("train score = ", score)


