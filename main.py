import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp

mp_holistic= mp.solutions.holistic
mp_drawing= mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #Image is not writable
    image.flags.writeable=False
    #For media pipe processing and make prediction
    results=model.process(image)
    image.flags.writeable=True
    image= cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results
#Definig Landmarks
def draw_landmarks(image,results):
    #mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1))
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
cap= cv2.VideoCapture(0)
#Access mediapipe model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame= cap.read()
        #Making Detection
        image, results= mediapipe_detection(frame, holistic)
        #Drawing Landmarks
        draw_landmarks(image, results)
        cv2.imshow('OpenCV Window', image)
        if cv2.waitKey(20) & 0xFF == ord('a'):
            break
    cap.release()
    cv2.destroyAllWindows() 
