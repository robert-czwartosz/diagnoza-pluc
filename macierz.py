#from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
import glob
import numpy as np
"""
cimport cython
from numpy cimport ndarray as ar
"""
import matplotlib.pyplot as plt
import cv2
import random
import itertools
from skimage.feature import greycomatrix, greycoprops
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from sklearn.metrics import classification_report

from itertools import chain



def quantizator(images, n_clusters):
    #(h, w) = image.shape[:2]
    # reshape the image into a feature vector so that k-means
    # can be applied
    # image = image.reshape((image.shape[0] * image.shape[1], 1))
    images = np.array(images).reshape((-1, 1))
     
    # apply k-means using the specified number of clusters and
    # then create the quantized image based on the predictions
    clt = MiniBatchKMeans(n_clusters = n_clusters)
    clt.fit(images)
    
    return clt

# %matplotlib inline
# =======================
# PRZYGOTOWANIE DANYCH
# ======================

root = ".\\chest-xray-pneumonia\\chest_xray\\"
IMG_DIM = (550, 550)
levels = 8

def read_images(files):
    print("START")
    print(len(files))
    random.shuffle(files)
    imgs = [cv2.resize(cv2.imread(img, cv2.IMREAD_GRAYSCALE), dsize=IMG_DIM) for img in files]
    imgs = [cv2.GaussianBlur(img,(5,5),0) for img in imgs]
    print("Images loaded and resized")
    labels = [fn.split('\\')[-2].strip() for fn in files]
    # encode text category labels
    le = LabelEncoder()
    le.fit(labels)
    labels_enc = le.transform(labels)
    labels_enc = np.array(labels_enc, dtype=float)
    return imgs, labels_enc, le.classes_

train_files = glob.glob(root+'train\\NORMAL\\*')[:200]
train_files_pneumonia = glob.glob(root+'train\\PNEUMONIA\\*')
random.shuffle(train_files_pneumonia)
train_files += train_files_pneumonia[:len(train_files)]
random.shuffle(train_files)
train_imgs, train_labels_enc, _ = read_images(train_files)

quantator = quantizator(train_imgs[:100], levels)
centers = quantator.cluster_centers_.flatten()
def minLevel():
    minimum = 1000
    index = 0
    for i in range(len(centers)):
        if minimum > centers[i]:
            index = i
            minimum = centers[i]
    return index

def data_processing(imgs):
    print("Quantization")
    imgs_quant = list(map(lambda x: quantator.predict(x.reshape(-1,1)).reshape(IMG_DIM), imgs))
    print("Images quantized")
    # Macierz wspolwystapien
    distances = [1]
    angles = [0]
    comats = [greycomatrix(img, distances, angles, levels=levels, symmetric=False, normed=True) for img in imgs_quant]
    print(comats[0][1:,1:,0,0])
    index = minLevel()
    #features = [np.delete(np.delete(np.array(comat), index, 1), index, 0).flatten() for comat in comats]
    features = [comat[1:,1:,:,:].flatten() for comat in comats]
    features = [list(map(lambda x: x/sum(feature), feature)) for feature in features]
    print(features[0])
    features = np.array(features)
    return features
    

train_features = data_processing(train_imgs)

# =======================
# UCZENIE SIECI
# ======================


clf = MLPClassifier(solver='adam', alpha=1e-5, learning_rate_init=0.001, learning_rate='adaptive', hidden_layer_sizes=(50, 2), random_state=1, max_iter=5000, verbose=True)  


# A sample toy binary classification dataset
def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]/len(y_true)
def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 1]/len(y_true)
def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 0]/len(y_true)
def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 1]/len(y_true)
scoring = {'tp': make_scorer(tp), 'tn': make_scorer(tn),
           'fp': make_scorer(fp), 'fn': make_scorer(fn),
           'accuracy': make_scorer(accuracy_score),
           'prec': make_scorer(precision_score),
           'recall': make_scorer(recall_score),
           'F1': make_scorer(f1_score)}
cv_results = cross_validate(clf, train_features, train_labels_enc, cv=5, scoring=scoring)
print(cv_results)

clf.fit(train_features, train_labels_enc)

# ======================================
# PRZYGOTOWANIE DANYCH TESTOWYCH
# ======================================

test_files = glob.glob(root+'test\\NORMAL\\*')
test_files_pneumonia = glob.glob(root+'test\\PNEUMONIA\\*')
random.shuffle(test_files_pneumonia)
test_files += test_files_pneumonia[:len(test_files)]
test_imgs, test_labels_enc, target_names = read_images(test_files)
test_features = data_processing(test_imgs)
# ======================================
# SPRAWDZENIE NA DANYCH TESTOWYCH
# ========================================

test_preds = clf.predict(test_features)
print(classification_report(test_labels_enc, test_preds, target_names=target_names))

