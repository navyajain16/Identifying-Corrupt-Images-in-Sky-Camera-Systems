# Identifying-Corrupt-Images-in-Sky-Camera-Systems
With the spirit of reproducible research, this repository contains all the codes required to produce the results in the manuscript:

> M. Jain, N. Jain, Y. H. Lee, and S. Dev, Identifying Corrupt Images in Sky Camera Systems (Under review)

### Executive summary
In this paper we have proposed a method to identify corrupt images in sky camera systems. In this method we have developed a framework which identifies the corrupt images and seperates them from the dataset.

Our framework is as follows:
1. First we calculate Entropy values, Laplacian values and Color channel values. 
2. These values are then used to train a machine learning model (In this experiment Decision tree, Random forest and K neighbours models are used)
3. The trained model is then used to classify images as corrupt or non-corrupt.

### Environment 
This project was tested on `python 3.8` using a `Windows 10` environment.

### Scripts
+ `Decision tree.py`: This file contains the code for training Decision tree model.
+ `Random forest.py`: This file contains the code for training Random forest model.
+ `KNN.py`: This file contains the code for training K neighbours model.
+ `Classification.py`: This file contains the code for classifying the images based on trained model.

