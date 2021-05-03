#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 20:10:12 2021

@author: alessiosavi


# DATASET CAN BE DOWNLOADED FROM THE FOLLOWING LINK:
    http://vis-www.cs.umass.edu/lfw/#download
"""

import tensorflow as tf
try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
except Exception as e:
    print("Unable to set memory grouwth!")
    print(e)
    pass

import os
import pickle
import sys

from sklearn.preprocessing import LabelEncoder

from utils import *


basedir = "lfw"
train_size = 0.8

# %% LOAD X AND Y
if os.path.exists("dataset/dataset_dlib_embedding.pkl"):
    with open("dataset/dataset_dlib_embedding.pkl", "rb") as f:
        X_train, X_test, y_train, y_test = pickle.load(f)
else:
    remove_person_with_few_images(basedir, 10)
    X_train, y_train = load_images(basedir, mode='train', train_size=train_size)
    X_test, y_test = load_images(basedir, mode='test', train_size=train_size)

assert len(X_train) == len(y_train) and len(y_train) > 0
assert len(X_test) == len(y_test) and len(y_test) > 0
# assert len(set(y_train)) == len(set(y_test)) and len(set(y_test)) > 0
# %%
label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(y_train + y_test)
y_train_label = label_encoder.transform(y_train)
y_test_label = label_encoder.transform(y_test)
mapping = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
sys.exit()


# %%


# %%


def estimator_input_fn(df_data, df_label, epochs=10, shuffle=True, batch_size=32):
    ds = tf.data.Dataset.from_tensor_slices((df_data, df_label))
    if shuffle:
        ds = ds.shuffle(100)
    ds = ds.batch(batch_size).repeat(epochs)
    return ds


train_input_fn = estimator_input_fn(X_train, y_train_label)
val_input_fn = estimator_input_fn(
    X_test, y_test_label, epochs=1, shuffle=False)
x_shape = None

for x, y in train_input_fn:
    x_shape = x.shape
    break
# %% CREATE MODEL
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir, histogram_freq=1)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=5)

model = tf.keras.Sequential(
    [
        layers.Dense(128, activation="relu", input_shape=(x_shape[1],)),
        layers.Dense(len(set(y_train)), activation="softmax"),
    ]
)
model.compile(
    optimizer="adam",
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
model.summary()

# %%
model.fit(train_input_fn, epochs=10, callbacks=[
    tensorboard_callback, early_stopping], validation_data=val_input_fn)
# %% EVALUATE MODEL
loss, acc = model.evaluate(val_input_fn)
print("Accuracy", acc)

# %% Extract not recognized photo

results = {}
for person in tqdm(os.listdir(basedir)[4:]):
    # Path related to all photos of a person
    person_path = pjoin(basedir, person)
    # All photos related to a person
    person_photos = os.listdir(person_path)
    for photo_path in tqdm(person_photos):
        img, face_locations = utils.extract_faces(pjoin(person_path, photo_path))
        for face in face_locations:
            face_encodings = utils.encodings(img, [face], utils.pose_predictor, utils.face_encoder)
            prediction = model.predict(face_encodings[0].reshape(1, -1))
            max_confidence_index = np.argmax(prediction)
            confidence = prediction[0][max_confidence_index]
            person_name = str(label_encoder.inverse_transform(
                np.array([max_confidence_index]))[0])
            result = {}
            result["confidence"] = confidence
            result["box"] = utils.rect_to_bb(face)
            results[person_name] = result
            if person != person_name:
                imsave(person_name + "_" + str(confidence) + ".jpg", img)
                imshow(utils.print_prediction_on_image(img, result))
