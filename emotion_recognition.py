import cv2
import numpy as np
import pandas as pd

#load dataset
dataset_path = '../emotion_dataset/fer2013/fer2013.csv'
img_size = (48, 48)


def load_dataset():
    data = pd.read_csv(dataset_path)
    pixels = data['pixels'].tolist()
    width, height = 48, 48
    faces = []
    for pixel in pixels:
        face = [int(px) for px in pixel.split(' ')]
        face = np.asarray(face).reshape(width, height)
        face = cv2.resize(face.astype('uint8'),img_size)
        faces.append(face.astype('float32'))
    faces = np.asarray(faces)
    faces = np.expand_dims(faces, 1)
    emotions = pd.get_dummies(data['emotion']).as_matrix()
    return faces, emotions

def preprocess_input(input_data, v2=True):
    input_data = input_data.astype('float32')
    input_data = input_data / 255.0 #scaled [0,1]
    if v2:
        input_data = (input_data - 0.5) * 2
    return input_data

faces, emotions = load_dataset()
faces = preprocess_input(faces)
xtrain, xtest, ytrain, ytest = trai