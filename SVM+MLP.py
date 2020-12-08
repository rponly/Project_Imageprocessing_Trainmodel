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
import tensorflow as tf
import keras
def image_to_feature_vector(image, size=(32, 32)):

	return cv2.resize(image, size).flatten()
def image_to_feature_vector_mnet(image, size=(160, 160)):

	return cv2.resize(image, size)

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



ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-p", "--Predictfeature", required=True,
	help="path to Predict feature")
ap.add_argument("-n", "--Name", required=True,
	help="path to user model")


args = vars(ap.parse_args())
imagePaths = list(paths.list_images(args["dataset"]))
testPaths = list(paths.list_images(args["Predictfeature"]))

try:
    os.mkdir(args["Name"])
except OSError:
    print ("Creation of the directory  failed")
else:
    print ("Successfully created the directory" )


folder_names = [
    './n0','./n1'
]
training_data = load_images_from_folder(folder_names)
#training_data = np.array(training_data)
from random import sample 

def under_sampling(min,data_list):
    expect_list = []
    new_list = []
    for t in data_list:
        new_list = sample(t,min)
        for n in new_list:
            expect_list.append(n)
        new_list = []
    return expect_list

min_images = min(number_of_images)
training_data = under_sampling(min_images,training_data)
""" training_data = tf.keras.applications.mobilenet_v2.preprocess_input(
    training_data, data_format=None) """

def defined_output(number_of_images):
    label = []
    for count,num in enumerate(number_of_images):
        for i in range(num):
            classes = len(number_of_images)
            output = np.zeros((classes,), dtype=int)
            output[count] = 1
            label.append(output)
    return label
number_of_images = []
for i in range(2): #2 class
    number_of_images.append(min_images)
y = defined_output(number_of_images)
y = np.array(y)
print(y)
#print(training_data.shape)
features = []
features_mnet =[]
labels = []
testfeatures = []
for (i, imagePath) in enumerate(imagePaths):
	# load the image and extract the class label (assuming that our
	# path as the format: /path/to/dataset/{class}.{image_num}.jpg
	image = cv2.imread(imagePath)
	label = imagePath.split(os.path.sep)[-1].split(".")[0]

	pixels = image_to_feature_vector(image)
	pixels_mnet = image_to_feature_vector_mnet(image)
	features.append(pixels)
	features_mnet.append(pixels_mnet)
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
features_mnet = np.array(features_mnet)
labels = np.array(labels)
#labels_mnet = defined_output(labels)
#labels_mnet = np.array(labels)
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
pickle.dump(svclassifier, open('./'+args["Name"]+ '/'+filename, 'wb'))
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
pickle.dump(mlp, open('./'+args["Name"]+ '/'+filename, 'wb'))
mlp_predict_result = mlp.predict(testfeatures)
print ("mlp predict")
print(mlp_predict_result)


#MoblienetV2
from keras.callbacks import ModelCheckpoint, EarlyStopping
filepath ='./'+args["Name"]+ '/'+'MV5_finalized_model.hdf5'
#cb = [EarlyStopping(monitor='val_loss', patience=2),
cb = [EarlyStopping(monitor='accuracy', patience=2),ModelCheckpoint(filepath=filepath, monitor='accuracy', save_best_only=True,mode='min',verbose=1)]


IMG_SIZE = 160
inputs = tf.keras.Input(shape=(160, 160, 3))
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False
X = base_model(inputs, training=False)
X = tf.keras.layers.GlobalAveragePooling2D()(X)
X = tf.keras.layers.Dropout(0.2)(X)
outputs = tf.keras.layers.Dense(2,activation='softmax')(X)
model = tf.keras.Model(inputs, outputs)

base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])
model.summary()
estimators.append(('MV5', model))
history = model.fit(features_mnet,y,epochs=10,callbacks=cb)
base_model.trainable = True



""" ensemble = VotingClassifier(estimators)
ensemble.fit(features, labels)
filename = 'voted_finalized_model.sav'
pickle.dump(ensemble, open(filename, 'wb'))
result = ensemble.predict(testfeatures)
print("Voted")
print(result) """
""" score_1 = ensemble.score(X_test, y_test)
print(score_1) """