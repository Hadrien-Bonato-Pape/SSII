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
from sklearn.metrics import confusion_matrix

def automatedTrainDataForSVM(c1, k1, toRecomputeKMEANS, verbose, minimal):
    if(minimal):
        print("c = ", c1)
        print("k = ", k1)

    #************************************************* INIT
        
    baryName = "./bary-logr-TP-fin/k" + str(k1)
    dimImg = []
    
    listImg=glob.glob(cat1+"/*.jpeg")
    tmp1 = len(listImg)
    groundTruth = [0] * tmp1
                                                                                                     
    listImg += glob.glob(cat2+"/*.jpeg")
    tmp2 = len(listImg) - tmp1
    groundTruth += [1] * tmp2

    listImg += glob.glob(cat3+"/*.jpg")
    tmp3 = len(listImg) - (tmp1 + tmp2)
    groundTruth += [2] * tmp3

    listImg += glob.glob(cat4+"/*.jpg")
    tmp4 = len(listImg) - (tmp1 + tmp2 + tmp3)
    groundTruth += [3] * tmp4
    
    lesSift = np.empty(shape=(0, 128), dtype=float)

    #************************************************* SIFT

    for s in listImg:
        image = cv2.imread(s)
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
        kp,des = sift.detectAndCompute(gray,None)
        dimImg.append(len(des))
        lesSift = np.append(lesSift,des,axis=0)

    #************************************************* KMEANS

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

    #************************************************* BOWS

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

    #************************************************* SVM

    if(verbose):
        print("SVM FIT")
    
    classif = svm.SVC(c1, kernel='rbf', decision_function_shape='ovo')
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

    #************************************************* INIT
    
    baryName = "./bary-logr-TP-fin/k" + str(k1)
    dimImg = []
    
    listImg=glob.glob(cat1+"/*.jpeg")
    tmp1 = len(listImg)
    groundTruth = [0] * tmp1
                                                                                                     
    listImg += glob.glob(cat2+"/*.jpeg")
    tmp2 = len(listImg) - tmp1
    groundTruth += [1] * tmp2

    listImg += glob.glob(cat3+"/*.jpg")
    tmp3 = len(listImg) - (tmp1 + tmp2)
    groundTruth += [2] * tmp3

    listImg += glob.glob(cat4+"/*.jpg")
    tmp4 = len(listImg) - (tmp1 + tmp2 + tmp3)
    groundTruth += [3] * tmp4
    
    lesSift = np.empty(shape=(0, 128), dtype=float)

    #************************************************* SIFT

    for s in listImg:
        image = cv2.imread(s)
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
        kp,des = sift.detectAndCompute(gray,None)
        dimImg.append(len(des))
        lesSift = np.append(lesSift,des,axis=0)

    #************************************************* KMEANS
           
    with open(baryName+'k.bary',  'rb') as input:
        km1 = pickle.load(input)

    bows = np.empty(shape=(0,k1),dtype=float)
    km1.predict(lesSift)
    
    #************************************************* BOWS
    
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

    #************************************************* SVM

    with open(baryName  + str(c1) +  'SVM.svm',  'rb') as input:
        classif = pickle.load(input)
        
    res = classif.predict(bows)
    print(res)
    print(groundTruth)
    f1Score = f1_score(groundTruth, res, average='macro')
    print("F1 score = ", f1Score)
    confu = confusion_matrix(groundTruth, res)
    print("Matrice de confusion = ", confu)
    print(" ")
    

#------------------------------------------------------------------------------------------------------------------------------------- Pseudo Main
toRecomputeKMEANS = False
verbose = True
minimal = True
c1 = 20
k1 = 50
f1Scores = []
workingDir = "./images-TP-fin/"



print("--------------------------------------------")
print("Training SVM")
cat1 = workingDir + "train/formule1"
cat2 = workingDir + "train/motoGrandPrix"
cat3 = workingDir + "train/chat4"
cat4 = workingDir + "train/chien3"
automatedTrainDataForSVM(c1, k1, toRecomputeKMEANS, verbose, minimal) #Training

cat1 = workingDir + "test/formule1"
cat2 = workingDir + "test/motoGrandPrix"
cat3 = workingDir + "test/chat4"
cat4 = workingDir + "test/chien3"
print("Validate with F1 score (with validate data -> pictures)")

automatedValidateDataSVM(c1, k1)

