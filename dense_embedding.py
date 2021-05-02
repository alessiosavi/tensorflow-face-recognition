#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 20:10:12 2021

@author: alessiosavi


# DATASET CAN BE DOWNLOADED FROM THE FOLLOWING LINK:
    http://vis-www.cs.umass.edu/lfw/#download
"""

# %%
import datetime
import pickle
from mtcnn import MTCNN
import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import dlib
from sklearn.preprocessing import LabelEncoder
import cv2
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import shutil
from os.path import join as pjoin
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
# %%

basedir = "lfw"
train_size = 0.8
face_detector = MTCNN()
n_transformation = 4

pose_predictor = dlib.shape_predictor(
    'models/shape_predictor_68_face_landmarks.dat')
face_encoder = dlib.face_recognition_model_v1(
    'models/dlib_face_recognition_resnet_model_v1.dat')

# %%
# extract a single face from a given photograph


def remove_person_with_few_images(basedir,  n):
    # Iterate every folder of dataset dir
    for person in tqdm(os.listdir(basedir)):
        # Path related to all photos of a person
        person_path = pjoin(basedir, person)
        # All photos related to a person
        # Avoid to manage person that have less than 10 photo
        if len(os.listdir(person_path)) < n:
            try:
                shutil.rmtree(person_path)
                print("Removing {} due to few than {} images ...".format(person, n))
            except OSError as e:
                print("Error: %s - %s." % (e.filename, e.strerror))


def extract_faces(filename="", image=None):
    faces = []
    # load image from file
    if image is None:
        image = cv2.imread(filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # detect faces in the image
    results = face_detector.detect_faces(image)

    for result in results:
        x, y, width, height = result["box"]
        faces.append(dlib.rectangle(x, y, x+width, y+height))
    return image, faces


def encodings(img, face_locations, pose_predictor, face_encoder):
    predictors = [pose_predictor(img, face_location)
                  for face_location in face_locations]
    return [np.array(face_encoder.compute_face_descriptor(img, predictor, 1)) for predictor in predictors]


# %% PREPROCESS
remove_person_with_few_images(basedir, 10)

# %% CREATE DATASET


datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=45,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)


# X_train = []
# y_train = []
# X_test = []
# y_test = []


def load_images(basedir, mode="train"):
    x, y = [], []
    # Iterate every folder of dataset dir
    for person in tqdm(os.listdir(basedir)):
        # Path related to all photos of a person
        person_path = pjoin(basedir, person)
        # All photos related to a person
        person_photos = os.listdir(person_path)
        if mode == 'train':
            start_index = 0
            stop_index = int(len(person_photos) * train_size)
        else:
            start_index = int(len(person_photos) * train_size)
            stop_index = len(person_photos)
        for photo_path in tqdm(person_photos[start_index:stop_index], leave=False, desc=person):
            img, face_locations = extract_faces(pjoin(person_path, photo_path))
            if len(face_locations) != 1:   # Avoid to manage photo where there are more than one face
                print("Skipping {} due to {} faces recognized".format(
                    pjoin(person_path, photo_path), len(face_locations)))
                os.remove(pjoin(person_path, photo_path))
                continue

            _, face_locations = extract_faces(image=img)
            face_encodings = encodings(
                img, face_locations, pose_predictor, face_encoder)
            if len(face_encodings) > 0:
                x.append(face_encodings[0])
                y.append(person)
    return x, y


# %% LOAD X AND Y

# X_train, y_train = load_images(basedir)
# X_test, y_test = load_images(basedir, mode='test')

with open("dataset/dataset_dlib_embedding.pkl","rb") as f:
    X_train, X_test, y_train,y_test= pickle.load(f)

assert len(X_train) == len(y_train) and len(y_train) > 0
assert len(X_test) == len(y_test) and len(y_test) > 0
# %%
label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(y_train)
y_train_label = label_encoder.transform(y_train)
y_test_label = label_encoder.transform(y_test)

# %%


def estimator_input_fn(df_data, df_label, epochs=10, shuffle=True, batch_size=32):
    ds = tf.data.Dataset.from_tensor_slices((df_data, df_label))
    if shuffle:
        ds = ds.shuffle(100)
    ds = ds.batch(batch_size).repeat(epochs)
    return ds


x_shape = None



train_input_fn = estimator_input_fn(X_train, y_train_label)
val_input_fn = estimator_input_fn(
    X_test, y_test_label, epochs=1, shuffle=False)

for x,y in train_input_fn:
    x_shape = x.shape
    break
# %% CREATE MODEL
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir, histogram_freq=1)


model = tf.keras.Sequential(
    [
        layers.Dense(128, activation="relu", input_shape=(x_shape[1],)),
        layers.Dense(len(set(y_train_label)), activation="softmax"),
    ]
)
model.compile(
    optimizer="adam",
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
model.summary()

# %%
model.fit(train_input_fn,  epochs=10, callbacks=[
          tensorboard_callback], validation_data=val_input_fn)
# %% EVALUATE MODEL
loss, acc = model.evaluate(test_x, y_test_label)
print("Accuracy", acc)