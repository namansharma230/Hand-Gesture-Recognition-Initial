import cv2
import numpy as np
import mediapipe as mp
from tensorflow.python.keras.models import load_model
import os
from matplotlib import pyplot as plt
import time
import keras
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import LSTM, Dense
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, LSTM, Dense, Flatten

# Load the saved model
loaded_model = load_model('action.h5')

# Initialize MediaPipe Holistic
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Function to perform mediapipe detection
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

# Defining Landmarks drawing functions
def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

# Defining styled Landmarks drawing function
def draw_styled_landmarks(image, results):
    landmarks_style = mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1)
    connections_style = mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)

    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, landmarks_style, connections_style)
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, landmarks_style, connections_style)

    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=1, circle_radius=1))
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=1, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=1, circle_radius=2))
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=1, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=1, circle_radius=2))

# Extracting key points function
def extract_key_points(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, face, lh, rh])

# Create CNN+LSTM model
num_landmarks = 1662
sequence_length = 30
actions = ["Hello", "Thanks", "I Like You"]
input_shape = (sequence_length, num_landmarks * 3)

# ... (previous imports)

from tensorflow.python.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Flatten

# ... (previous code)

# Function to create the CNN+LSTM model
def create_cnn_lstm_model(input_shape):
    model = Sequential()

    # CNN layers
    model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(128, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())

    # LSTM layers
    model.add(LSTM(64, return_sequences=True, activation='relu'))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))

    # Dense layers
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(len(actions), activation='softmax'))

    return model


# Compile the model
load_model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# OpenCV video capture
cap = cv2.VideoCapture(0)

# Main loop for real-time detection
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()

        # Perform detection using the loaded_model
        image, results = mediapipe_detection(frame, holistic)
        draw_styled_landmarks(image, results)
        keypoints = extract_key_points(results)

        # Reshape the keypoints to match the model's input_shape
        keypoints = keypoints.reshape((1, sequence_length, num_landmarks * 3))

        # Use the loaded model for detection
        predictions = loaded_model.predict(keypoints)
        detected_action = actions[np.argmax(predictions)]

        # Display the result on the frame
        cv2.putText(image, f'Detected Action: {detected_action}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('OpenCV Window', image)

        # Break the loop if 'a' key is pressed
        if cv2.waitKey(20) & 0xFF == ord('a'):
            break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
