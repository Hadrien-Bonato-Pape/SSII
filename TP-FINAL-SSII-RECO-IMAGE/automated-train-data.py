import glob
from sys import argv
import cv2
import pickle

import shutil
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import f1_score
 
cat1 = "/Users/hbp/Documents/GitHub/SSII/TP-FINAL-SSII-RECO-IMAGE/images3/train/formule1"
cat2 = "/Users/hbp/Documents/GitHub/SSII/TP-FINAL-SSII-RECO-IMAGE/images3/train/motoGrandPrix"

def automatedTrainData(k1):
    print("--------------------------------------------")
    print("k = ", k1)
    baryName = "/Users/hbp/Documents/GitHub/SSII/TP-FINAL-SSII-RECO-IMAGE/bary-logr-images3/k" + str(k1)

    listImg=glob.glob(cat1+"/*.jpeg")
    tmpa = len(listImg)
    listImg += glob.glob(cat2+"/*.jpeg")
                   
    lesSift = np.empty(shape=(0, 128), dtype=float) # array of all SIFTS from all images
    dimImg = [] # nb of sift per file
    groundTruth = [0]*tmpa # result we would like to reach
    tmpb = len(listImg)-tmpa
    groundTruth += [1]*tmpb

    print("SIFT")
    for s in listImg:
        image = cv2.imread(s)
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
        kp,des = sift.detectAndCompute(gray,None)
        dimImg.append(len(des))
        lesSift = np.append(lesSift,des,axis=0)
           
    #BOW initialization
    bows = np.empty(shape=(0,k1),dtype=float)

    # everything ready for the k-means
    print("KMEAN FIT")
    kmeans1 = KMeans(n_clusters=k1, random_state=0).fit(lesSift)

    with open(baryName+'k.bary', 'wb') as output:
        pickle.dump(kmeans1, output, pickle.HIGHEST_PROTOCOL)

    bary1 = kmeans1.cluster_centers_;
        
    #writing the BOWs for second k-means
    print("BOWS")
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
    print("Regression logistique FIT")
    logisticRegr = LogisticRegression(max_iter=2000)
    logisticRegr.fit(bows, groundTruth)
    with open(baryName+'L.logr', 'wb') as output:
        pickle.dump(logisticRegr, output, pickle.HIGHEST_PROTOCOL)

    score = logisticRegr.score(bows, groundTruth)
    print("train score = ", score)
    print(" ")

automatedTrainData(5)
automatedTrainData(7)
automatedTrainData(10)
automatedTrainData(11)
automatedTrainData(12)
automatedTrainData(13)
automatedTrainData(14)
automatedTrainData(15)
automatedTrainData(16)
automatedTrainData(17)
automatedTrainData(18)
automatedTrainData(19)
automatedTrainData(20)
automatedTrainData(22)
automatedTrainData(25)
automatedTrainData(30)
automatedTrainData(40)
automatedTrainData(50)
automatedTrainData(100)
automatedTrainData(200)
