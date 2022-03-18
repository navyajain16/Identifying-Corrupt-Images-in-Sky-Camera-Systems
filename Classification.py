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
src ="C:\dataset cloud\test"
#Path for directory created for storing corrupt images
dub ="C:\dataset cloud\bad"
 
corimg=[]

#Creating labels
Categories =['badimg','goodimg']

for category in Categories:
        label = Categories.index(category)

#Images passed through various steps of the framework
for image in os.listdir(src): 
        cloud_img = cv2.imread(image)
        shawl_gray =rgb2gray(cloud_img)
        result = shannon_entropy(shawl_gray[:,0]) # Finding Entropy 
        fm = variance_of_laplacian(shawl_gray)  #Finding variance of Laplacian
        #Any one of the following colour channels are to be used, in this code we have appended grayscale channel values.
        valuess = shawl_gray.var() #Finding grayscale values
        red = cloud_img[:,:,2] #Finding Red channel values
        redvalue = red.var()
        green = cloud_img[:,:,1] #Finding Green channel values
        greenvalue= green.var()
        blue = cloud_img[:,:,0] #Finding Blue Channel values
        bluevalue= blue.var()
        redblue = cloud_img[:,:,2] - cloud_img[:,:,0] #Finding R-B channel values
        redblueval = redblue.var()
        redss = cloud_img[:,:,2]/ (cloud_img[:,:,0] + 0.1) #Finding R/B channel values
        cc = redss.max()
        dd = redss.min()
        redsso =(redss-cc)*(225/(dd-cc))
        redbvalue = redsso.var()
        if redbvalue and redblueval == 0:
            rbvalue = 0
            print ("value 0")
        else:
            rb = (cloud_img[:,:,0] - cloud_img[:,:,2])/(cloud_img[:,:,0] + cloud_img[:,:,2]+0.1) 
            c= np.amax(rb)
            d= rb.min() 
            rbo = (rb-c)*(225/(d-c)) 
            rbvalue = rbo.var() #Finding R-B/R+B channel values
        hsv_img = cv2.cvtColor(cloud_img, cv2.COLOR_BGR2HSV)
        h = hsv_img[:, :, 0]
        hval = h.var() #Finding h channel values
        s = hsv_img[:, :, 1] #Finding s channel values
        sval = s.var()
        v = hsv_img[:, :, 2] #Finding v channel values
        vval = v.var()
        img = np.vstack(np.asarray([result, fm,valuess])) #Appending the values of entropy, laplacian and one color channel (in this case, grayscale)
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
for value,img in zip(ypred,corimg):
    if value == 0:
        print(img + " " + "bad")
        shutil.move(os.path.join(src, img), dub)
    else:
        print("good")

