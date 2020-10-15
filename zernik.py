# https://towardsdatascience.com/a-comprehensive-hands-on-guide-to-transfer-learning-with-real-world-applications-in-deep-learning-212bf3b2f27a
import glob
import numpy as np
import cv2
import random
from skimage.feature import greycomatrix, greycoprops
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
#from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img

from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from sklearn.metrics import classification_report

import mahotas
# =======================
# MOMENTY ZERNIKE'A
# ======================
class ZernikeMoments:
    def __init__(self, radius, degree):
        # store the size of the radius that will be
        # used when computing moments
        self.radius = radius
        self.degree = degree

    def describe(self, image):
        # return the Zernike moments for the image
        return mahotas.features.zernike_moments(image, self.radius, self.degree)


# initialize descriptor (Zernike Moments with a radius
# of 21 used to characterize the shape)
desc = ZernikeMoments(100, 11)
thresh = 100

# =======================
# PRZYGOTOWANIE DANYCH
# ======================

root = ".\\chest-xray-pneumonia\\chest_xray\\"
IMG_DIM = (550, 550)

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

train_files = glob.glob(root+'train\\NORMAL\\*')
train_files_pneumonia = glob.glob(root+'train\\PNEUMONIA\\*')
random.shuffle(train_files_pneumonia)
train_files += train_files_pneumonia[:len(train_files)]
random.shuffle(train_files)
train_imgs, train_labels_enc, _ = read_images(train_files)

def data_processing(imgs):
    print("Moments")
    moments = [desc.describe(img) for img in imgs]
    print("Moments calculated")
    features = [moment.flatten() for moment in moments]
    print("Flatten")
    features = np.array(features)
    print("To np.array converted")
    print(features[0])
    return features
    

train_features = data_processing(train_imgs)


# =====================================
# UCZENIE SIECI I WALIDACJA KRZYŻOWA
# ========================================


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

