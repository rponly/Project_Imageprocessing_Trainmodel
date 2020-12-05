import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import pickle
import argparse
from os.path import basename
from imutils import paths
def image_to_feature_vector(image, size=(32, 32)):

	return cv2.resize(image, size).flatten()
#path
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--Predictfeature", required=True,
	help="path to Predict feature")
args = vars(ap.parse_args())
testPaths = list(paths.list_images(args["Predictfeature"]))
testfeatures = []
for (i, testPaths) in enumerate(testPaths):
	# load the image and extract the class label (assuming that our
	# path as the format: /path/to/dataset/{class}.{image_num}.jpg
	image = cv2.imread(testPaths)
	

	pixels = image_to_feature_vector(image)
	testfeatures.append(pixels)
testfeatures = np.array(testfeatures)


filename = 'SVM_finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
SVM_result = loaded_model.predict(testfeatures)
print(SVM_result)

filename = 'MLP_finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
MLP_result = loaded_model.predict(testfeatures)
print ("mlp predict")
print(MLP_result)

filename = 'KNN_finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
KNN_result = loaded_model.predict(testfeatures)
print ("knn predict")
print(KNN_result)