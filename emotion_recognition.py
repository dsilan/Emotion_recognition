
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split 

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
    emotions = pd.get_dummies(data['emotion']).values
    return faces, emotions

def preprocess_input(input_data, v2=True):
    input_data = input_data.astype('float32')
    input_data = input_data / 255.0 #scaled [0,1]
    if v2:
        input_data = (input_data - 0.5) * 2 #scaled  [-1,1]
    return input_data

faces, emotions = load_dataset()
faces = preprocess_input(faces)
xtrain, xtest, ytrain, ytest = train_test_split(faces, emotions, test_size=0.2, shuffle=True)

from keras.preprocessing.image import img_to_array
from keras.models import load_model
import imutils

 
# parameters for loading data and images
detection_model_path = './model/haarcascade_frontalface_default.xml'
emotion_model_path = './model/_mini_XCEPTION.106-0.65.hdf5'
img_path = "../img/mark.png"
 
# hyper-parameters for bounding boxes shape
# loading models
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry","disgust","scared", "happy", "sad", "surprised","neutral"]
 
#reading the frame
orig_frame = cv2.imread(img_path)
frame = cv2.imread(img_path,0)
faces = face_detection.detectMultiScale(frame,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
 
if len(faces) > 0:
    faces = sorted(faces, reverse=True,key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
    (fX, fY, fW, fH) = faces
    roi = frame[fY:fY + fH, fX:fX + fW]
    roi = cv2.resize(roi, (48, 48))
    roi = roi.astype("float") / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis=0)
    preds = emotion_classifier.predict(roi)[0]
    emotion_probability = np.max(preds)
    label = EMOTIONS[preds.argmax()]
    cv2.putText(orig_frame, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    cv2.rectangle(orig_frame, (fX, fY), (fX + fW, fY + fH),(0, 0, 255), 2)
 
cv2.imshow('test_face', orig_frame)
cv2.imwrite('test_output/'+img_path.split('/')[-1],orig_frame)
cv2.waitKey(2000)


cv2.destroyAllWindows()
