import argparse
import imutils
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from IPython.display import display
from matplotlib import style
import pandas as pd
import numpy as np
from PIL import Image
from skimage.color import rgb2grey
from sklearn.metrics import accuracy_score
import matplotlib.cm as cm
import math
from scipy import ndimage
import functools
from os.path import basename
from imutils import paths
import pickle
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import VotingClassifier
from sklearn.utils.estimator_checks import check_estimator
def image_to_feature_vector(image, size=(32, 32)):

	return cv2.resize(image, size).flatten()


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-p", "--Predictfeature", required=True,
	help="path to Predict feature")
args = vars(ap.parse_args())
imagePaths = list(paths.list_images(args["dataset"]))
testPaths = list(paths.list_images(args["Predictfeature"]))
features = []
labels = []
testfeatures = []
for (i, imagePath) in enumerate(imagePaths):
	# load the image and extract the class label (assuming that our
	# path as the format: /path/to/dataset/{class}.{image_num}.jpg
	image = cv2.imread(imagePath)
	label = imagePath.split(os.path.sep)[-1].split(".")[0]

	pixels = image_to_feature_vector(image)
	
	features.append(pixels)
	labels.append(label)

	# show an update every 1,000 images
	if i > 0 and i % 1000 == 0:
		print("[INFO] processed {}/{}".format(i, len(imagePaths)))

for (i, testPaths) in enumerate(testPaths):
	# load the image and extract the class label (assuming that our
	# path as the format: /path/to/dataset/{class}.{image_num}.jpg
	image = cv2.imread(testPaths)
	

	pixels = image_to_feature_vector(image)
	testfeatures.append(pixels)
	
# show some information on the memory consumed by the raw images
# matrix and features matrix
features = np.array(features)
labels = np.array(labels)
#testfeatures = np.array(testfeatures)
estimators = []

#SVM
from sklearn.svm import SVC
svclassifier = SVC(kernel='linear')
params_svm = {"kernel":"rbf", "C":10, "gamma":0.000001}
svclassifier.set_params(**params_svm)
estimators.append(('svm', svclassifier))
svclassifier.fit(features, labels)
filename = 'SVM_finalized_model.sav'
pickle.dump(svclassifier, open(filename, 'wb'))
SVM_predict_result = svclassifier.predict(testfeatures)
score = svclassifier.score
print ("svm predict")
print(SVM_predict_result)
print(score)

#MLP
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(88,), activation='relu', solver='lbfgs', max_iter=500 ,random_state=42) #48 72 80 r 70
estimators.append(('mlp', mlp))
mlp.fit(features,labels)
filename = 'MLP_finalized_model.sav'
pickle.dump(mlp, open(filename, 'wb'))
mlp_predict_result = mlp.predict(testfeatures)
print ("mlp predict")
print(mlp_predict_result)

#KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
estimators.append(('knn', knn))
knn.fit(features, labels)
filename = 'KNN_finalized_model.sav'
pickle.dump(knn, open(filename, 'wb'))
knn_predict_result = knn.predict(testfeatures)
print ("knn predict")
print(knn_predict_result)


ensemble = VotingClassifier(estimators)
ensemble.fit(features, labels)
result = ensemble.predict(testfeatures)
print("Voted")
print(result)
""" score_1 = ensemble.score(X_test, y_test)
print(score_1) """