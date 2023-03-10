import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
from keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import pandas as pd
import cv2
from IPython.display import clear_output as cls
import imutils
import zipfile

# Modeling
import tensorflow as tf
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, GlobalAveragePooling2D as GAP
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint

# Visualization
import plotly.express as px
import matplotlib.pyplot as plt
from keras.applications import ResNet50V2
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout, MaxPool2D, Input, ReLU
from keras.applications import ResNet50V2
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV


def preprocess_image(image_address):
    image_or = cv2.imread(image_address)
    image_rs = cv2.resize(image_or, (128, 128))
    image_blr = cv2.GaussianBlur(image_rs, (5, 5), 0)
    image_gray = cv2.cvtColor(image_blr, cv2.COLOR_BGR2GRAY)
    return (image_gray)


aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")
from google.colab import drive

drive.mount('/content/drive')
os.chdir('drive/MyDrive/latest final')
train_dir = 'Training'
sub_list = os.listdir(train_dir)
for subd in sub_list:
    subd_path = os.path.join(train_dir, subd)
    if os.path.isdir(subd_path):
        image_l = os.listdir(subd_path)
        for image_name in image_l:
            ig_address = os.path.join(subd_path, image_name)
            img_pr = preprocess_image(ig_address)
            # replacing the old image in directory
            cv2.imwrite(os.path.join(subd_path, image_name), img_pr)
train_dat = aug.flow_from_directory('Training', target_size=(128, 128), shuffle=True, batch_size=16)
test_dir = 'Testing'
sub_list = os.listdir(test_dir)
for subd in sub_list:
    subd_path = os.path.join(test_dir, subd)
    if os.path.isdir(subd_path):
        image_l = os.listdir(subd_path)
        for image_name in image_l:
            ig_address = os.path.join(subd_path, image_name)
            img_pr = preprocess_image(ig_address)
            # replacing the old image in directory
            cv2.imwrite(os.path.join(subd_path, image_name), img_pr)
test_dat = aug.flow_from_directory('Testing', target_size=(128, 128), shuffle=True, batch_size=16)
model1 = Sequential()
model1.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(128, 128, 3)))
model1.add(Conv2D(32, kernel_size=(3, 3), activation="relu"))
model1.add(MaxPool2D(pool_size=(2, 2)))
model1.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
model1.add(MaxPool2D(pool_size=(2, 2)))
model1.add(Dropout(0.2))
# model.add(BatchNormalization())
model1.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
model1.add(MaxPool2D(pool_size=(2, 2)))
model1.add(Dropout(0.2))

model1.add(Flatten())
model1.add(Dense(32, activation="relu"))
model1.add(Dense(64, activation="relu"))
model1.add(Dense(32, activation="relu"))
model1.add(Dense(4, activation="softmax"))

model1.summary()
model1.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
trained_model1 = model1.fit(train_dat, validation_data=test, epochs=20)
accur = trained_model1.history['accuracy']
val_accur = trained_model1.history['val_accuracy']
epochs = range(len(accur))
fig = plt.figure(figsize=(14, 7))
plt.plot(epochs, accur, 'r', label="Training Accuracy")
plt.plot(epochs, val_accur, 'b', label="Validation Accuracy")
plt.legend(loc="lower right")
plt.show()
train_loss = trained_model1.history['loss']
train_v_loss = trained_model1.history['val_loss']
epochs = range(len(train_loss))
fig = plt.figure(figsize=(14, 7))
plt.plot(epochs, train_loss, 'r', label="Training loss")
plt.plot(epochs, train_v_loss, 'b', label="Validation loss")
plt.legend(loc="upper right")
plt.show()
base_mod = ResNet50V2(input_shape=(128, 128, 3), include_top=False)
base_mod.trainable = False

# defining our model
n = "ResNet50V2"
model3 = Sequential([
    base_mod,
    GAP(),
    Dense(256, activation='relu', kernel_initializer='he_normal'),
    Dense(4, activation='softmax'),
], name=n)

# compiling model
model3.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# training model
trained_model3 = model3.fit(train_dat, validation_data=test_dat, epochs=20)
accur3 = trained_model3.history['accuracy']
val_accur3 = trained_model3.history['val_accuracy']
epochs = range(len(accur3))
fig = plt.figure(figsize=(14, 7))
plt.plot(epochs, accur3, 'r', label="Training Accuracy")
plt.plot(epochs, val_accur3, 'b', label="Validation Accuracy")
plt.legend(loc="lower right")
plt.show()
train_loss3 = trained_model3.history['loss']
train_v_loss3 = trained_model3.history['val_loss']
epochs = range(len(train_loss3))
fig = plt.figure(figsize=(14, 7))
plt.plot(epochs, train_loss3, 'r', label="Training loss")
plt.plot(epochs, train_v_loss3, 'b', label="Validation loss")
plt.legend(loc="upper right")
plt.show()
model2 = Sequential()
model2.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)))
model2.add(Conv2D(64, (3, 3), activation='relu'))
model2.add(MaxPooling2D(2, 2))
model2.add(Dropout(0.3))
model2.add(Conv2D(64, (3, 3), activation='relu'))
model2.add(Conv2D(64, (3, 3), activation='relu'))
model2.add(Dropout(0.3))
model2.add(MaxPooling2D(2, 2))
model2.add(Dropout(0.3))
model2.add(Conv2D(128, (3, 3), activation='relu'))
model2.add(Conv2D(128, (3, 3), activation='relu'))
model2.add(Conv2D(128, (3, 3), activation='relu'))
model2.add(MaxPooling2D(2, 2))
model2.add(Dropout(0.3))
model2.add(Conv2D(128, (3, 3), activation='relu'))
model2.add(Conv2D(256, (3, 3), activation='relu'))
model2.add(MaxPooling2D(2, 2))
model2.add(Dropout(0.3))

model2.add(Flatten())
model2.add(Dense(512, activation='relu'))
model2.add(Dense(512, activation='relu'))
model2.add(Dropout(0.3))
model2.add(Dense(4, activation='softmax'))
model2.summary()
model2.compile(loss="categorical_crossentropy", optimizer="Adam", metrics=["accuracy"])
trained_model2 = model2.fit(train_dat, validation_data=test_dat, epochs=20)
