#Face detection with openCV and deep learning
import numpy as np
import argparse
import cv2

from keras.preprocessing.image import img_to_array
from keras.models import load_model
import imutils
import sys
emotion_model_path = 'model/_mini_XCEPTION.106-0.65.hdf5'
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry","disgust","scared", "happy", "sad", "surprised","neutral"]

ap = argparse.ArgumentParser()
ap.add_argument("-i","--image", required=True, help="path to input image")
ap.add_argument("-p","--prototxt", required=True, help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m","--model", required=True, help="path to Caffe pre-trained model")
ap.add_argument("-c","--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

image = cv2.imread(args["image"])
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

# pass the blob through the network and obtain the detections and predictions
print("[INFO] computing object detections...")
net.setInput(blob) 
detections = net.forward()

#loop over the detections
for i in range(0, detections.shape[2]):
    #extract the confidence (probability) associated with the prediction
    confidence = detections[0, 0, i, 2]

    # filter out weak detections by ensuring the `confidence` is
	# greater than the minimum confidence
    if confidence > args["confidence"]:
        #compute the (x, y)-coordinates of the bounding box for the object
        box = detections[0,0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        roi = image[startY:endY, startX:endX]
        roi = cv2.resize(roi, (48, 48))
        roi = roi[:,:,0]
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        preds = emotion_classifier.predict(roi)[0]
        emotion_probability = np.max(preds)
        label = EMOTIONS[preds.argmax()]
        cv2.putText(image, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.rectangle(image, (startX, startY), (endX, endY),(0, 0, 255), 2)

        """
        # draw the bounding box of the face along with the associated probability
        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
        cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        """
cv2.imshow('test_face', image)
#cv2.imwrite('test_output/'+img_path.split('/')[-1],orig_frame)
cv2.waitKey(8000)
cv2.destroyAllWindows()
#show the output image
#cv2.imshow("Output", image)
#cv2.waitKey(0)
