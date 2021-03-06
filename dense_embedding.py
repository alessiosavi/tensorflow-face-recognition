#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 20:10:12 2021

@author: alessiosavi


# DATASET CAN BE DOWNLOADED FROM THE FOLLOWING LINK:
    http://vis-www.cs.umass.edu/lfw/#download
    
The version that is the model is tuned against is filtered using the follwing condition:
    - At least 10 photos for a person
    - Only the face that contains a single face of the target person
"""
# from line_profiler_pycharm import profile
import tensorflow as tf

# Use the following env
#export TF_XLA_FLAGS=--tf_xla_enable_xla_devices
#export TF_FORCE_GPU_ALLOW_GROWTH='true'
#export LA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda
#export TF_ENABLE_AUTO_MIXED_PRECISION=1

import datetime
import os
import pickle
from tqdm import tqdm
from os.path import join as pjoin


import numpy as np
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder
from matplotlib.pyplot import imsave
import utils


basedir = "lfw"
train_size = 0.8

# %% LOAD X AND Y if already parsed
if os.path.exists("dataset/dataset_dlib_embedding.pkl"):
    with open("dataset/dataset_dlib_embedding.pkl", "rb") as f:
        X_train, X_test, y_train, y_test = pickle.load(f)
else:
    # Remove not usable target (that contains lower than 10 photos)
    utils.remove_person_with_few_images(basedir, 10)
    X_train, y_train = utils.load_images(
        basedir, mode='train', train_size=train_size)
    X_test, y_test = utils.load_images(
        basedir, mode='test', train_size=train_size)
    with open("dataset/dataset_dlib_embedding.pkl", "wb") as f:
        pickle.dump((X_train, X_test, y_train, y_test), f)

assert len(X_train) == len(y_train) and len(y_train) > 0
assert len(X_test) == len(y_test) and len(y_test) > 0
assert len(set(y_train)) == len(set(y_test)) and len(set(y_test)) > 0
# %%
label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(y_train + y_test)
y_train_label = label_encoder.transform(y_train)
y_test_label = label_encoder.transform(y_test)
mapping = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))

# %%

EPOCHS = 10
# SHUFFLE_BUFFER = 100
BATCH_SIZE = 32
PREFETCH_BUFFER = 8


def estimator_input_fn(df_data, df_label, train=True):
    ds = tf.data.Dataset.from_tensor_slices((df_data, df_label))
    if train:
        ds = ds.repeat(EPOCHS)
        # ds = ds.shuffle(SHUFFLE_BUFFER, seed=0)
    ds = ds.batch(BATCH_SIZE)
    if train:
        ds = ds.prefetch(PREFETCH_BUFFER)
    options = tf.data.Options()
    options.experimental_deterministic = not train
    options.experimental_optimization.autotune = True
    options.experimental_optimization.autotune_buffers = True
    options.experimental_optimization.filter_fusion = True
    # options.experimental_optimization.hoist_random_uniform = True  ## Cannot be set on tensorflow 2.6.0
    options.experimental_optimization.map_and_batch_fusion = True
    options.experimental_optimization.map_and_filter_fusion = False
    options.experimental_optimization.map_fusion = True
    options.experimental_optimization.map_parallelization = True
    # options.experimental_optimization.map_vectorization.enabled = True  ## Cannot be set on tensorflow 2.6.0
    # options.experimental_optimization.map_vectorization.use_choose_fastest = True  ## Cannot be set on tensorflow 2.6.0
    options.experimental_optimization.noop_elimination = True
    options.experimental_optimization.parallel_batch = True
    # options.experimental_optimization.shuffle_and_repeat_fusion = True
    options.experimental_optimization.apply_default_optimizations = False
    options.threading.max_intra_op_parallelism = 1
    options.threading.private_threadpool_size = 8
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.AUTO
    ds = ds.with_options(options)
    return ds


train_input_fn = estimator_input_fn(X_train, y_train_label)
val_input_fn = estimator_input_fn(X_test, y_test_label, train=False)
x_shape = None

for x, y in train_input_fn:
    x_shape = x.shape
    break
# %% CREATE MODEL
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir, histogram_freq=1)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=10)

model = tf.keras.Sequential(
    [
        layers.Dense(128, activation="relu", input_shape=(x_shape[1],)),
        # layers.Dense(64, activation="relu"),
        # layers.Dropout(0.2),
        layers.Dense(len(set(y_train)), activation="softmax"),
    ]
)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=["accuracy"],
)
model.summary()

# %%
model.fit(train_input_fn, epochs=1000, callbacks=[tensorboard_callback, early_stopping], validation_data=val_input_fn)
# %% EVALUATE MODEL
loss, acc = model.evaluate(val_input_fn)
print("Accuracy", acc)
# %% Extract not recognized photo
results = {}

for person in tqdm(os.listdir(basedir)):
    # Path related to all photos of a person
    person_path = utils.pjoin(basedir, person)
    # All photos related to a person
    person_photos = os.listdir(person_path)
    for photo_path in person_photos:
        img, face_locations = utils.extract_faces(pjoin(person_path, photo_path))
        for face in face_locations:
            face_encodings = utils.encodings(img, [face], utils.pose_predictor, utils.face_encoder)
            prediction = model.predict(face_encodings[0].reshape(1, -1))
            max_confidence_index = np.argmax(prediction)
            confidence = prediction[0][max_confidence_index]
            person_name = str(label_encoder.inverse_transform(np.array([max_confidence_index]))[0])
            result = {}
            result["confidence"] = confidence
            result["box"] = utils.rect_to_bb(face)
            results[person_name] = result
            if person != person_name:
                image_with_box = utils.print_prediction_on_image(img, result)
                imsave("not_classified/"+person + "|" + person_name + str(confidence) + ".jpg", image_with_box)

# %%

def predict(model, img):
    img, face_locations = utils.extract_faces(image=img)
    face_encodings = utils.encodings(
        img, [face], utils.pose_predictor, utils.face_encoder)
    prediction = model.predict(face_encodings[0].reshape(1, -1))
    max_confidence_index = np.argmax(prediction)
    confidence = prediction[0][max_confidence_index]
    person_name = str(label_encoder.inverse_transform(
        np.array([max_confidence_index]))[0])
    result = {}
    result["confidence"] = confidence
    result["box"] = utils.rect_to_bb(face)
    result["person"] = person_name
    return result


# img, _ = utils.extract_faces(pjoin(person_path, photo_path))
# print(predict(model, img))
