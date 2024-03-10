import cv2
import mediapipe as mp
import numpy as np
import pickle

def perform_hand_gesture_recognition(model_path, data_txt, confidence_threshold=0.7):
    # Load the model
    model_dict = pickle.load(open(model_path, 'rb'))
    model = model_dict['model']

    cap = cv2.VideoCapture(0)

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)

    # Load class labels from data.txt
    with open(data_txt, 'r') as f:
        labels_dict = {line.split()[0]: line.split()[1] for line in f.readlines()}

    while True:
        data_aux = []
        x_ = []
        y_ = []

        ret, frame = cap.read()

        H, W, _ = frame.shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

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
            max_sequence_length = 55  # Adjust the length based on your needs
            data_aux = data_aux[:max_sequence_length] + [0] * max(0, max_sequence_length - len(data_aux))

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10

            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            prediction = model.predict([np.asarray(data_aux)])

            numeric_class_label = str(prediction[0])
            prediction_confidence = model.predict_proba([np.asarray(data_aux)])[0].max()
            predicted_class_name = labels_dict.get(numeric_class_label, 'Unknown')

            print("Predicted Class Label (Numeric):", numeric_class_label)
            print("Predicted Class Name:", predicted_class_name)
            print("Prediction Confidence:", prediction_confidence)

            if prediction_confidence < confidence_threshold:
                predicted_class_name = 'Unknown'
                numeric_class_label = "?"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, f"{numeric_class_label} {predicted_class_name}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

        cv2.imshow('frame', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Example usage:
# model_path = 'model\hand_gesture_model.p'
# perform_hand_gesture_recognition(model_path, 'model\hand_gesture_model.txt')
