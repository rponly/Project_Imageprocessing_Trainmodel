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
from sklearn.ensemble import ExtraTreesClassifier
def image_to_feature_vector(image, size=(32, 32)):

	return cv2.resize(image, size).flatten()


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
args = vars(ap.parse_args())
imagePaths = list(paths.list_images(args["dataset"]))

features = []
labels = []

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

# show some information on the memory consumed by the raw images
# matrix and features matrix
features = np.array(features)
labels = np.array(labels)

#SVM
from sklearn.svm import SVC
svclassifier = SVC(kernel='linear')
params_svm = {"kernel":"rbf", "C":10, "gamma":0.000001}
svclassifier.set_params(**params_svm)
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
mlp.fit(features,labels)
filename = 'MLP_finalized_model.sav'
pickle.dump(mlp, open(filename, 'wb'))
mlp_predict_result = mlp.predict(testfeatures)
print ("mlp predict")
print(mlp_predict_result)