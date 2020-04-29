import glob
from sys import argv
import cv2
import pickle

import shutil
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import f1_score

cat1 = "/Users/hbp/Documents/GitHub/SSII/TP-FINAL-SSII-RECO-IMAGE/images3/validation/formule1"
cat2 = "/Users/hbp/Documents/GitHub/SSII/TP-FINAL-SSII-RECO-IMAGE/images3/validation/motoGrandPrix"

def automatedValidateData(k1):
    baryName = "/Users/hbp/Documents/GitHub/SSII/TP-FINAL-SSII-RECO-IMAGE/bary-logr-images3/k" + str(k1)
    
    listImg=glob.glob(cat1+"/*.jpeg")
    tmpa = len(listImg)
    listImg += glob.glob(cat2+"/*.jpeg")
                   
    lesSift = np.empty(shape=(0, 128), dtype=float) # array of all SIFTS from all images
    dimImg = [] # nb of sift per file
    groundTruth = [0]*tmpa # result we would like to reach
    tmpb = len(listImg)-tmpa
    groundTruth += [1]*tmpb

    for s in listImg:
        image = cv2.imread(s)
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
        kp,des = sift.detectAndCompute(gray,None)
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

    print("k = " + str(k1))
    print(res)
    print(groundTruth)

    score = logisticRegr.score(bows, groundTruth)
    print("f1 score = ",f1_score(groundTruth, res))
    print("train score = ", score)


automatedValidateData(5)
automatedValidateData(7)
automatedValidateData(10)
automatedValidateData(11)
automatedValidateData(12)
automatedValidateData(13)
automatedValidateData(14)
automatedValidateData(15)
automatedValidateData(16)
automatedValidateData(17)
automatedValidateData(18)
automatedValidateData(19)
automatedValidateData(20)
automatedValidateData(22)
automatedValidateData(25)
automatedValidateData(30)
automatedValidateData(40)
automatedValidateData(50)
automatedValidateData(100)
automatedValidateData(200)
