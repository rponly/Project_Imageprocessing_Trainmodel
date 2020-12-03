from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import pickle
#path
testfeatures = []

Featureimage =cv2.imread(path)
testimage = image_to_feature_vector(Featureimage)
testfeatures.append(testimage)
testfeatures = np.array(testfeatures)

filename = 'SVM_finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
SVM_result = loaded_model.predict(testfeatures)
print(result)

filename = 'MLP_finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
MLP_result = loaded_model.predict(testfeatures)
print ("mlp predict")
print(MLP_result)