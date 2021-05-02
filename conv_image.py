#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 20:10:12 2021

@author: alessiosavi
"""

# %%
import glob
from multiprocessing.pool import ThreadPool
import datetime
# from mtcnn import MTCNN
import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import dlib
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import cv2
from tensorflow.keras import layers
import shutil
from os.path import join as pjoin
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
# %%
basedir = "/opt/SP/workspace/JupyterLab/Tensorflow-Certification/FaceRecognition/lfw"
train_size = 0.8
# face_detector = MTCNN()
img_height, img_width = 224, 224
batch_size = 32

# %%


# %%


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


def resize_image(img, h, w):
    return cv2.resize(img, dsize=(h, w), interpolation=cv2.INTER_LANCZOS4)


def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = abs(rect.left())
    y = abs(rect.top())
    w = abs(rect.right() - x)
    h = abs(rect.bottom() - y)

    # return a tuple of (x, y, w, h)
    return (x, y, w, h)
# %%


X_train = []
y_train = []
X_test = []
y_test = []


def remove_folder(path):
    try:
        shutil.rmtree(path)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))


def clean_folders_core(person_path, photo_path, person):
    # print("person_path, photo_path, person:", person_path, photo_path, person)
    _, face_locations = extract_faces(pjoin(person_path, photo_path))
    if len(face_locations) != 1:   # Avoid to manage photo where there are more than one face
        print("\nRemoving {} due to {} faces recognized".format(
            pjoin(person_path, photo_path), len(face_locations)))
        os.remove(pjoin(person_path, photo_path))
        if len(os.listdir(person_path)) < 10:
            remove_folder(person_path)
            print("Removing {} due to low images ...", person)


def clean_folders(basedir):
    for person in tqdm(os.listdir(basedir)):
        # Path related to all photos of a person
        person_path = pjoin(basedir, person)
        # All photos related to a person
        person_photos = os.listdir(person_path)

        # Avoid to manage person that have less than 10 photo
        if len(person_photos) < 10:
            print("Removing {} due to {} images ...",
                  person, len(person_photos))
            remove_folder(person_path)
            continue
        pool = ThreadPool()
        for photo_path in person_photos:
            # clean_folders_core(person_path, photo_path, person)
            pool.apply_async(clean_folders_core, args=(
                person_path, photo_path, person))
        pool.close()
        pool.join()
# %%


# clean_folders(basedir)


train_dir = pjoin(basedir, 'train')
val_dir = pjoin(basedir, 'val')

# Iterate lfw/george_bush
for person in os.listdir(basedir):
    person_photo_path = pjoin(basedir, person)
    person_photos = os.listdir(person_photo_path)
    if len(person_photos) < 10:
        continue
    num_train = int(round(len(person_photos)*0.8))

    train, val = person_photos[:num_train], person_photos[num_train:]
    print("{}: {} Images".format(person, len(person_photos)))

    for photo_path in train:
        if not os.path.exists(os.path.join(basedir, 'train', person)):
            os.makedirs(os.path.join(basedir, 'train', person))
        shutil.copy(pjoin(person_photo_path, photo_path),
                    os.path.join(basedir, 'train', person, photo_path))

    for photo_path in val:
        if not os.path.exists(os.path.join(basedir, 'val', person)):
            os.makedirs(os.path.join(basedir, 'val', person))
        shutil.copy(pjoin(person_photo_path, photo_path),
                    os.path.join(basedir, 'val', person, photo_path))


# %% CREATE DATASET

# train_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     basedir,
#     validation_split=0.2,
#     subset="training",
#     seed=123,
#     image_size=(224, 224),
#     batch_size=batch_size)

# val_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     basedir,
#     validation_split=0.2,
#     subset="validation",
#     seed=123,
#     image_size=(224, 224),
#     batch_size=batch_size)

# AUTOTUNE = tf.data.AUTOTUNE

# train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
# val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=30,
    zoom_range=0.3,
    fill_mode="nearest",
    horizontal_flip=True,
    vertical_flip=True,
    rescale=1./255,
    validation_split=0.0,
    dtype=None,
)


train_data_gen = datagen.flow_from_directory(batch_size=batch_size,
                                             directory=train_dir,
                                             shuffle=True,
                                             target_size=(
                                                img_height, img_width),
                                             class_mode='sparse')

image_gen_val = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

val_data_gen = image_gen_val.flow_from_directory(
    batch_size=batch_size, directory=val_dir, target_size=(img_height, img_width), class_mode='sparse')

# %%
URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
feature_extractor = hub.KerasLayer(URL,input_shape=(img_height, img_width, 3))
feature_extractor.trainable = False
# %% CREATE MODEL
log_dir = "/opt/SP/workspace/JupyterLab/Tensorflow-Certification/FaceRecognition/logs/conv_image/" + \
    datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True)

n_classes = len(val_data_gen.class_indices)

model = tf.keras.Sequential([
  feature_extractor,
  layers.Dense(n_classes)
])

# model = tf.keras.Sequential(
#     [
#         # layers.experimental.preprocessing.Rescaling(1./255, input_shape=(224, 224, 3)),
#         # tf.keras.layers.experimental.preprocessing.RandomFlip(),
#         # tf.keras.layers.experimental.preprocessing.RandomRotation(.3),
#         # tf.keras.layers.experimental.preprocessing.RandomZoom(.2),

#         # layers.Conv2D(64, kernel_size=(3, 3), activation='relu',input_shape=(img_height, img_width, 3)),
#         # layers.MaxPooling2D(pool_size=(2, 2)),
#         # layers.Dropout(0.3),

#         # layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
#         # layers.MaxPooling2D(pool_size=(2, 2)),
#         # layers.Dropout(0.3),


#         # layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
#         # layers.MaxPooling2D(pool_size=(2, 2)),
#         # layers.Dropout(0.3),
#         feature_extractor,

#         # layers.Conv2D(n_classes, kernel_size=(3, 3),
#         #               padding='same', activation='relu'),
#         # layers.GlobalAveragePooling2D(),

#         # tf.keras.applications.VGG16(weights=None,pooling="max",
#         #     classes=n_classes, input_shape=(224, 224, 3)),

#         # layers.Flatten(),
#         # layers.Dense(256, activation="relu"),
#         layers.Dense(n_classes, activation="softmax")
#     ]
# )


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    # optimizer='adam',
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)
model.summary()


# %% FIT MODEL
model.fit(train_data_gen, validation_data=val_data_gen,
          epochs=500, callbacks=[tensorboard_callback])

# %% EVALUATE MODEL
loss, acc = model.evaluate(val_data_gen)
print("Accuracy", acc)