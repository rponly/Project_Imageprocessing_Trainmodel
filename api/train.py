# libraries import
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
from random import sample 

# utility functions
def image_to_feature_vector(image, size=(32, 32)):
	return cv2.resize(image, size).flatten()

def image_to_feature_vector_mnet(image, size=(160, 160)):
	return cv2.resize(image, size)

# global variables

def load_images_from_folder(folders):
    all_images = []
    number_of_images = []
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
    return all_images, number_of_images

def under_sampling(min, data_list):
    expect_list = []
    new_list = []
    for t in data_list:
        new_list = sample(t,min)
        for n in new_list:
            expect_list.append(n)
        new_list = []
    return expect_list

def defined_output(number_of_images):
    label = []
    for count,num in enumerate(number_of_images):
        for i in range(num):
            classes = len(number_of_images)
            output = np.zeros((classes,), dtype=int)
            output[count] = 1
            label.append(output)
    return label

def train_model(model_name):
    folder_names = ["./" + model_name + "/gen", "./" + model_name + "/fake"]
    training_data, number_of_images = load_images_from_folder(folder_names)
    min_images = min(number_of_images)
    training_data = under_sampling(min_images, training_data)
    pathGen = list(paths.list_images("./" + model_name + "/gen"))
    pathFake = list(paths.list_images("./" + model_name + "/fake"))
    imagePaths = pathGen + pathFake
    # imagePaths = list(paths.list_images('Dataset_Kaggle'))
    # print(imagePaths)
    # make the number of those classes are same amount (minimum amount)
    number_of_images = []
    for i in range(2): # 2 classes
        number_of_images.append(min_images)

    y = defined_output(number_of_images)
    y = np.array(y)
    print('y is', y)
    # print(training_data.shape)
    features = []
    features_mnet =[]
    labels = []

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
        if i > 0 and i % 10 == 0:
            print("[INFO] processed {}/{}".format(i, len(imagePaths)))

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
    pickle.dump(svclassifier, open('./'+model_name+ '/'+filename, 'wb'))
    
    #MLP
    from sklearn.neural_network import MLPClassifier
    mlp = MLPClassifier(hidden_layer_sizes=(88,), activation='relu', solver='lbfgs', max_iter=500 ,random_state=42) #48 72 80 r 70
    #estimators.append(('mlp', mlp))

    mlp.fit(features,labels)
    filename = 'MLP_finalized_model.sav'
    pickle.dump(mlp, open('./'+model_name+ '/'+filename, 'wb'))

    #MoblienetV2
    from keras.callbacks import ModelCheckpoint, EarlyStopping
    filepath ='./'+model_name+ '/'+'MV5_finalized_model.hdf5'
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

    print('shape of x is ', features_mnet.shape)
    print('shape of y is ', y.shape)

    history = model.fit(features_mnet,y,epochs=10,callbacks=cb)
    base_model.trainable = True