import os
import shutil
from os.path import join as pjoin

import cv2
import dlib
import numpy as np
from mtcnn import MTCNN
from tqdm import tqdm

face_detector = MTCNN()

pose_predictor = dlib.shape_predictor(
    'models/shape_predictor_68_face_landmarks.dat')
face_encoder = dlib.face_recognition_model_v1(
    'models/dlib_face_recognition_resnet_model_v1.dat')


def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return x, y, w, h


def print_prediction_on_image(img, predictions):
    """
    Shows the face recognition results visually.
    :param img: path to image to be recognized
        np.array
    :param predictions: results of the predict function
        dict -> {'George_HW_Bush': {'confidence': 0.999826, 'box': (83, 63, 105, 121)}}
    :return:
    """
    clone = img.copy()
    for name in predictions:
        (x, y, w, h) = predictions[name]["box"]
        # Draw a box around the face using the Pillow module
        # draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))
        cv2.rectangle(clone, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.putText(clone, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.putText(clone, str(predictions[name]["confidence"]), (
            x, y + h + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # Display the resulting image
    return clone


def remove_person_with_few_images(basedir, n=10):
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
        faces.append(dlib.rectangle(x, y, x + width, y + height))
    return image, faces


# %%
# extract a single face from a given photograph


def encodings(img, face_locations, pose_predictor, face_encoder):
    predictors = [pose_predictor(img, face_location)
                  for face_location in face_locations]
    return [np.array(face_encoder.compute_face_descriptor(img, predictor, 1)) for predictor in predictors]


# %% CREATE DATASET


def load_images(basedir, mode="train", train_size=0.8):
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
        for photo_path in tqdm(person_photos[start_index:stop_index]):
            img, face_locations = extract_faces(pjoin(person_path, photo_path))
            if len(face_locations) != 1:  # Avoid to manage photo where there are more than one face
                print("Skipping {} due to {} faces recognized".format(
                    pjoin(person_path, photo_path), len(face_locations)))
                os.remove(pjoin(person_path, photo_path))
            else:
                face_encodings = encodings(
                    img, face_locations, pose_predictor, face_encoder)
                if len(face_encodings) > 0:
                    x.append(face_encodings[0])
                    y.append(person)
    return x, y
