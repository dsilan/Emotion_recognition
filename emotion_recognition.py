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
    faces
