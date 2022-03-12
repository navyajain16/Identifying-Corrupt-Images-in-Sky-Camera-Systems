# -*- coding: utf-8 -*-
"""
Created on Sat Aug 14 00:54:26 2021

@author: navya
"""
import os 
import numpy as np
import cv2
import matplotlib.pyplot as plt 
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import shutil 
from sklearn import metrics
from skimage.color import rgb2hsv, rgb2gray, rgb2yuv
from skimage.measure.entropy import shannon_entropy
import shutil
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import glob
import time

start = time.time()

def variance_of_laplacian(image):
        return cv2.Laplacian(image, cv2.CV_64F).var()

data=[]

pick = open('modeclouds.sav', 'rb')
model = pickle.load(pick)
pick.close() 


Categories =['badimg','goodimg']

src ="test"
dub ="bad"
 
catimg=[]

for category in Categories:
        label = Categories.index(category)
        
for image in os.listdir(src): 
        cloud_img = cv2.imread(image)
        #gray = cv2.cvtColor(cloud_img, cv2.COLOR_BGR2GRAY)
        shawl_gray =rgb2gray(cloud_img)
        result = shannon_entropy(shawl_gray[:,0])
        #imagei = cv2.imread(image)
        fm = variance_of_laplacian(shawl_gray) 
        valuess = shawl_gray.var()
        img = np.vstack(np.asarray([result, fm,valuess]))
        jimage = np.array(cloud_img).flatten()
        finalimg = np.array([jimage,img])
        fimg = finalimg.flatten() 
        data.append([fimg,label])  
        catimg.append(image)
        #print(data)
print(len(data))
features= [] 
labels = [] 
for feature,label in data:
        features.append(feature) 
        labels.append(label)
images = np.asarray((list(sub[1] for sub in features))) 
images=images.reshape(len(images),3) 
probability=model.predict_proba(images)
ypred = model.predict(images)
#print(probability)
dec = np.asarray((list(sub[0] for sub in probability)))
for value,img in zip(ypred,catimg):
    if value == 0:
        print(img + " " + "bad")
        shutil.move(os.path.join(src, img), dub)
    else:
        print("good")
        
stop = time.time()
print(stop-start)
