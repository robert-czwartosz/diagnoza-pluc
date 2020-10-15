
import cv2
import glob
import random
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split

# =======================
# PRZYGOTOWANIE DANYCH
# ======================

root = ".\\chest-xray-pneumonia\\chest_xray\\"
IMG_DIM = (300, 300)

def read_images(files):
    print("START")
    print(len(files))
    random.shuffle(files)
    imgs = [cv2.resize(cv2.imread(img, cv2.IMREAD_GRAYSCALE), dsize=IMG_DIM) for img in files]
    imgs = np.array(imgs)
    imgs = imgs.reshape(len(files), 300, 300,1)
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

train_imgs, validation_imgs, train_labels_enc, validation_labels_enc = train_test_split(
    train_imgs, train_labels_enc, test_size=0.2, random_state=123321)


print('Train dataset shape:', train_imgs.shape, 
      '\tValidation dataset shape:', validation_imgs.shape)


train_imgs_scaled = train_imgs.astype('float32')
validation_imgs_scaled  = validation_imgs.astype('float32')
train_imgs_scaled /= 255
validation_imgs_scaled /= 255

print(train_imgs[0].shape)
#array_to_img(train_imgs[0])




batch_size = 30
num_classes = 2
epochs = 4
input_shape = (300, 300, 1)




# ===========================
# MODEL
# =================
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers

model = Sequential()

model.add(Conv2D(16, kernel_size=(4, 4), activation='relu', 
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(4, 4)))

model.add(Conv2D(64, kernel_size=(4, 4), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, kernel_size=(4, 4), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.8))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))


model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(),
              metrics=['accuracy'])

model.summary()



# =================
# TRENING
# ==================
history = model.fit(x=train_imgs_scaled, y=train_labels_enc,
                    validation_data=(validation_imgs_scaled, validation_labels_enc),
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1)

model.save('Konwolucje.h5')


# =================
# WIZUALIZACJA
# =================
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
t = f.suptitle('Basic CNN Performance', fontsize=12)
f.subplots_adjust(top=0.85, wspace=0.3)

epoch_list = list(range(1,epochs+1))
ax1.plot(epoch_list, history.history['accuracy'], label='Train Accuracy')
ax1.plot(epoch_list, history.history['val_accuracy'], label='Validation Accuracy')
ax1.set_xticks(np.arange(0, epochs+1, 5))
ax1.set_ylabel('Accuracy Value')
ax1.set_xlabel('Epoch')
ax1.set_title('Accuracy')
l1 = ax1.legend(loc="best")

ax2.plot(epoch_list, history.history['loss'], label='Train Loss')
ax2.plot(epoch_list, history.history['val_loss'], label='Validation Loss')
ax2.set_xticks(np.arange(0, epochs+1, 5))
ax2.set_ylabel('Loss Value')
ax2.set_xlabel('Epoch')
ax2.set_title('Loss')
l2 = ax2.legend(loc="best")

f.show()


# ==========================================
# SPRAWDZENIE NA DANYCH TESTOWYCH
# =========================================

test_files = glob.glob(root+'test\\NORMAL\\*')
test_files_pneumonia = glob.glob(root+'test\\PNEUMONIA\\*')
random.shuffle(test_files_pneumonia)
test_files += test_files_pneumonia[:len(test_files)]
test_imgs, test_labels_enc, target_names = read_images(test_files)

test_imgs_scaled = test_imgs.astype('float32')
test_imgs_scaled /= 255

test_preds = model.predict(
    test_imgs_scaled,
    batch_size=None,
    verbose=0,
    steps=None,
    callbacks=None,
    max_queue_size=10,
    workers=1,
    use_multiprocessing=False
)
test_preds = (test_preds > 0.5)*1
print(classification_report(test_labels_enc, test_preds, target_names=target_names))
print("Accuracy: "+str(np.sum(test_labels_enc==test_preds.flatten())/test_labels_enc.size))