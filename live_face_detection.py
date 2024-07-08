# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 17:46:17 2024

@author: Dhrumit Patel
"""

from keras.models import load_model
from time import sleep
from keras_preprocessing.image import img_to_array
from keras_preprocessing import image
import cv2

import numpy as np

face_classifier = cv2.CascadeClassifier('pretrained_haarcascade_classifier/haarcascade_frontalface_default.xml')
emotion_model = load_model('models/emotion_detection_model_50epochs.h5')
age_model = load_model('models/age_model_3epochs.h5')
gender_model = load_model('models/gender_model_3epochs.h5')

class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
gender_labels = ['Male', 'Female']

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    labels = []
    
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_classifier.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray=gray[y:y+h,x:x+w]
        roi_gray=cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)

        # Get image ready for prediction
        roi=roi_gray.astype('float')/255.0  # Scaling the image
        roi=img_to_array(roi)
        roi=np.expand_dims(roi,axis=0)  # Expand dims to get it ready for prediction (1, 48, 48, 1)

        preds=emotion_model.predict(roi)[0]  # One hot encoded result for 7 classes
        label=class_labels[preds.argmax()]  # Find the label
        label_position=(x,y)
        cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        
        #Gender
        roi_color=frame[y:y+h,x:x+w]
        roi_color=cv2.resize(roi_color,(200,200),interpolation=cv2.INTER_AREA)
        gender_predict = gender_model.predict(np.array(roi_color).reshape(-1,200,200,3))
        gender_predict = (gender_predict>= 0.5).astype(int)[:,0]
        gender_label=gender_labels[gender_predict[0]] 
        gender_label_position=(x,y+h+50) # 50 pixels below to move the label outside the face
        cv2.putText(frame,gender_label,gender_label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        
        #Age
        age_predict = age_model.predict(np.array(roi_color).reshape(-1,200,200,3))
        age = round(age_predict[0,0])
        age_label_position=(x+h,y+h)
        cv2.putText(frame,"Age="+str(age),age_label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        
   
    cv2.imshow('Live Face Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

































