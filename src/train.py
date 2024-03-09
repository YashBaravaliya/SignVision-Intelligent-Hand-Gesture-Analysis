import os
import pickle
import mediapipe as mp
import cv2
from sklearn.ensemble import RandomForestClassifier
import numpy as np


def train_hand_gesture_model(data_path, model_folder, model_name):
    mp_hands = mp.solutions.hands
    mp_drawing_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

    data = []
    labels = []
    max_sequence_length = 55  # Adjust the length based on your needs

    for dir_ in os.listdir(data_path):
        for img_path in os.listdir(os.path.join(data_path, dir_)):
            data_aux = []

            x_ = []
            y_ = []

            img = cv2.imread(os.path.join(data_path, dir_, img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            results = hands.process(img_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y

                        x_.append(x)
                        y_.append(y)

                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))

                # Pad or truncate the data_aux sequence to a fixed length
                data_aux = data_aux[:max_sequence_length] + [0] * max(0, max_sequence_length - len(data_aux))

                data.append(data_aux)
                labels.append(dir_)

    # Train the model
    model = RandomForestClassifier()
    data_array = np.asarray(data)
    labels_array = np.asarray(labels)

    model.fit(data_array, labels_array)

    # Save the model to the specified folder
    model_path = os.path.join(model_folder, model_name)

    # Check if the file already exists and overwrite it
    if os.path.exists(model_path):
        print(f"Model file {model_path} already exists. Overwriting...")
        os.remove(model_path)

    os.makedirs(model_folder, exist_ok=True)
    with open(model_path, 'wb') as model_file:
        pickle.dump({'model': model}, model_file)

    print(f'Model saved successfully at: {model_path}')


# Example usage:
# data_folder = 'data\\test'
# model_folder = 'model'
# model_name = 'hand_gesture_model.p'

# train_hand_gesture_model(data_folder, model_folder, model_name)
