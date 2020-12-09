import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import pickle
import argparse
from os.path import basename
from imutils import paths
import keras
import tensorflow as tf
import os
from random import sample 
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import VotingClassifier
from sklearn.utils.estimator_checks import check_estimator
estimators = []
number_of_images = []
def load_images_from_folder(folders):
    all_images = []
    images = []
    count = 0
    for folder in folders:
        for filename in os.listdir(folder):
            img = cv2.imread(os.path.join(folder,filename))
            b,g,r = cv2.split(img)       # get b,g,r
            img = cv2.merge([r,g,b])     # switch it to rgb
            if img is not None:
                img = cv2.resize(img,(160,160))
                images.append(img)
                count = count + 1
        number_of_images.append(count) #นับว่าแต่ละ directory มีกี่รูป
        all_images.append(images)
        count = 0
        images = []
    return all_images

def image_to_feature_vector(image, size=(32, 32)):

	return cv2.resize(image, size).flatten()
def defined_output(number_of_images):
    label = []
    for count,num in enumerate(number_of_images):
        for i in range(num):
            classes = len(number_of_images)
            output = np.zeros((classes,), dtype=int)
            output[count] = 1
            label.append(output)
    return label
#path
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--Predictfeature", required=True,
	help="path to Predict feature")
ap.add_argument("-n", "--Name", required=True,
	help="path to user model")
args = vars(ap.parse_args())
testPaths = list(paths.list_images(args["Predictfeature"]))
print(args["Predictfeature"])
folder_names = [args["Predictfeature"]
]
training_data = load_images_from_folder(folder_names)
training_data = np.array(training_data)
""" d,e,ma,b,c=training_data
training_data=(a,b,c) """
testfeatures = []
testother = []
for (i, testPaths) in enumerate(testPaths):
	# load the image and extract the class label (assuming that our
	# path as the format: /path/to/dataset/{class}.{image_num}.jpg
	image = cv2.imread(testPaths)
	pixels = image_to_feature_vector(image)
	b,g,r = cv2.split(image)       # get b,g,r
	image = cv2.merge([r,g,b]) 
	image = cv2.resize(image,(160,160))

	
	testfeatures.append(image)
	testother.append(pixels)
testfeatures = np.array(testfeatures)
testother = np.array(testother)

filename = 'SVM_finalized_model.sav'
loaded_model1 = pickle.load(open('./'+args["Name"]+ '/'+filename, 'rb'))
SVM_result = loaded_model1.predict(testother)
estimators.append(('svm', loaded_model1))


filename = 'MLP_finalized_model.sav'
loaded_model2 = pickle.load(open('./'+args["Name"]+ '/'+filename, 'rb'))
MLP_result = loaded_model2.predict(testother)
estimators.append(('MLP', loaded_model2))




filename = 'MV5_finalized_model.hdf5'
loaded_model3 = tf.keras.models.load_model('./'+args["Name"]+ '/'+filename)
MV5_result = loaded_model3.predict(testfeatures)
estimators.append(('MV5', loaded_model3))


print(SVM_result)
print(MLP_result)
print(MV5_result)

from mlxtend.classifier import EnsembleVoteClassifier
from sklearn.preprocessing import LabelEncoder
import copy
#eclf = EnsembleVoteClassifier(clfs=[loaded_model1, loaded_model2, loaded_model3],refit=False)
""" clf_list = [loaded_model1, loaded_model2, loaded_model3]
eclf = VotingClassifier(estimators = [('1' ,loaded_model1), ('2', loaded_model2), ('3', loaded_model3)], voting='soft')

eclf.estimators_ = clf_list
eclf.le_ = LabelEncoder().fit([0])
eclf.classes_ = eclf.le_.classes_
nsamples, nx, ny ,nz = testfeatures.shape
testfeatureee = testfeatures.reshape((nsamples,nx*ny*nz))
print(testfeatureee.shape)
eclf.predict(testfeatureee)
print(result) """
""" ensemble = VotingClassifier(estimators)
ensemble.fit()
result = ensemble.predict(testfeatures) """