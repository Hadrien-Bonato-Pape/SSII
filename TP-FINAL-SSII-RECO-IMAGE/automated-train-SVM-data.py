from sklearn import svm
import glob
from sys import argv
import cv2
import pickle
import shutil
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import f1_score

def automatedTrainDataForSVM(c1, k1, toRecomputeKMEANS, verbose, minimal):
    if(minimal):
        print("c = ", c1)
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

    for s in listImg:
        image = cv2.imread(s)
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
        kp,des = sift.detectAndCompute(gray,None)
        dimImg.append(len(des))
        lesSift = np.append(lesSift,des,axis=0)

    if(toRecomputeKMEANS):
        if(verbose):
            print("KMEAN FIT")
        km1 = KMeans(n_clusters=k1, random_state=0).fit(lesSift)
        with open(baryName+'k.bary', 'wb') as output:
            pickle.dump(km1, output, pickle.HIGHEST_PROTOCOL)
    else:
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

    #ready for the SVM

    if(verbose):
        print("SVM FIT")
    
    classif = svm.SVC(c1,kernel='rbf')
    classif.fit(bows, groundTruth)
    
    with open(baryName + str(c1) + 'SVM.svm', 'wb') as output:
        pickle.dump(classif, output, pickle.HIGHEST_PROTOCOL)
                    
    if(verbose):
        predict = classif.predict(bows)
        vectors = classif.support_vectors_
        score = classif.score(bows, groundTruth)
        print('Prediction class for bows', predict)
        print('Support vectors: ', vectors)
        print("Train score = ", score)

    if(minimal):
        print(" ")


def automatedValidateDataSVM(c1, k1):
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

    with open(baryName  + str(c1) +  'SVM.svm',  'rb') as input:
        classif = pickle.load(input)


    res = classif.predict(bows)
    print(res)
    print(groundTruth)
    f1Score = f1_score(groundTruth, res)
    print("F1 score = ", f1Score)
    print(" ")
    return f1Score

#------------------------------------------------------------------------------------------------------------------------------------- Pseudo Main
toRecomputeKMEANS = False
verbose = False
minimal = True
c1 = 0.1
k1 = 25
f1Scores = []
interval = 0.1
maxC1 = 1

while(c1 < maxC1):
    print("--------------------------------------------")
    print("Training SVM")
    cat1 = "/Users/hbp/Documents/GitHub/SSII/TP-FINAL-SSII-RECO-IMAGE/images3/train/formule1"
    cat2 = "/Users/hbp/Documents/GitHub/SSII/TP-FINAL-SSII-RECO-IMAGE/images3/train/motoGrandPrix"
    automatedTrainDataForSVM(c1, k1, toRecomputeKMEANS, verbose, minimal) #Training
    cat1 = "/Users/hbp/Documents/GitHub/SSII/TP-FINAL-SSII-RECO-IMAGE/images3/validation/formule1"
    cat2 = "/Users/hbp/Documents/GitHub/SSII/TP-FINAL-SSII-RECO-IMAGE/images3/validation/motoGrandPrix"
    print("Validate with F1 score (with validate data -> pictures)")
    f1Scores.append(automatedValidateDataSVM(c1, k1))
    c1 = c1 + interval

maxF1 = max(f1Scores)
i = 0
intervalRecherche = interval
print("--------------------------------------------")
print("Find best F1 SCORE : ")
print("Best F1 SCORE = ", maxF1)
print("For values of C : ")

while(intervalRecherche < maxC1):
    if(f1Scores[i] == maxF1):
        print(intervalRecherche)
    i = i + 1
    intervalRecherche = intervalRecherche + interval

print("All values of F1")
for s in f1Scores:
    print(s)
