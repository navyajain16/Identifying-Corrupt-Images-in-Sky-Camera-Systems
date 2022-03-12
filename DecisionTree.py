import os 
import numpy as np
import cv2
import matplotlib.pyplot as plt 
import pickle
import random
from sklearn.model_selection import train_test_split
import shutil 
from sklearn import metrics
from skimage.color import rgb2hsv, rgb2gray, rgb2yuv
from skimage.measure.entropy import shannon_entropy
import shutil
from sklearn.tree import DecisionTreeClassifier

src="C:\dataset cloud\test"  #Path of inital dataset
des ="C:\dataset cloud\bad"  #Path of repository created for storing corrupt images 

data = []

categories = ['badimg', 'goodimg'] #Creating labels

#Function for Variance of Laplacian
def variance_of_laplacian(image):
        return cv2.Laplacian(image, cv2.CV_64F).var()
  
#Storing the data in form of array
for category in categories:
    path = os.path.join(src, category)
    label = categories.index(category)
    for image in os.listdir(path):
        imgpath = os.path.join(path, image)
        cloud_img = cv2.imread(imgpath,0)
        shawl_gray =rgb2gray(cv2.imread(imgpath)) 
        result = shannon_entropy(shawl_gray[:,0]) 
        fm = variance_of_laplacian(shawl_gray) 
        valuess = shawl_gray.var()
        img = np.vstack([result, fm, valuess])
        jimage = np.array(cloud_img).flatten()
        finalimg = np.array([jimage,img])
        fimg = finalimg.flatten()
        data.append([fimg,label])
print(len(data))

pick_in = open('datacloudss.pickle', 'wb')
pickle.dump(data,pick_in)
pick_in.close()
       
pick_in = open('datacloudss.pickle', 'rb')
data = pickle.load(pick_in)
pick_in.close()

#Shuffling  the data
random.shuffle(data)
features= []
labels = []

for feature,label in data:
    features.append(feature)
    labels.append(label)

#Splitting the data into train and test dataset
xtrain, xtest, ytrain, ytest = train_test_split(features,labels, test_size=0.20)

#Importing Decision Tree Classifier
clf = DecisionTreeClassifier(criterion = "entropy", random_state = 50)

#Modifying data to be used in training the model 
xtrain = np.asarray((list(sub[1] for sub in xtrain)))
print(xtrain.shape)
xtrain= xtrain.reshape(len(xtrain),3)
print(xtrain.shape)
len(xtrain)

# Train Decision Tree Classifer
clf = clf.fit(xtrain,ytrain)
 
#Saving the model 
pick = open('modeclouds.sav', 'wb')
pickle.dump(clf, pick)
pick.close() 
print("dumped")

x_test = np.asarray((list(sub[1] for sub in xtest)))
x_test=x_test.reshape(len(x_test),3)
#Predict the response for test dataset
ypred = clf.predict(x_test)

#Finding the accuracy of the model
print("Accuracy:",metrics.accuracy_score(ytest, ypred))

#Printing the confusion matrix and classification report
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(ytest, ypred))
print(classification_report(ytest, ypred))

#Predict the response for test dataset
for test, pred in zip(xtest, ypred):
    print("Test Value: ",test, " Pred Value: ",pred)
    
