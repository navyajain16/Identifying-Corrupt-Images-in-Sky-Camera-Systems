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
import pandas as pd
import glob

#Defining function for Laplacian
def variance_of_laplacian(image):
        return cv2.Laplacian(image, cv2.CV_64F).var()

data=[]

#Loading the trained model
pick = open('modeclouds.sav', 'rb')
model = pickle.load(pick)
pick.close() 

#Path for source dataset 
src ="test"
#Path for directory created for storing corrupt images
dub ="bad"
 
corimg=[]

#Creating labels
Categories =['badimg','goodimg']

for category in Categories:
        label = Categories.index(category)

#Images passed through various steps of the framework
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
        corimg.append(image)
        
print(len(data))
features= [] 
labels = [] 

#Modifying the data to be predicted by the model
for feature,label in data:
        features.append(feature) 
        labels.append(label)
images = np.asarray((list(sub[1] for sub in features))) 
images=images.reshape(len(images),3) 

#Predicting the data
probability=model.predict_proba(images)
ypred = model.predict(images)
dec = np.asarray((list(sub[0] for sub in probability)))

#Moving corrupt images to seperate directory
for value,img in zip(ypred,catimg):
    if value == 0:
        print(img + " " + "bad")
        shutil.move(os.path.join(src, img), dub)
    else:
        print("good")

