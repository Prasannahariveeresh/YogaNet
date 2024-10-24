import cv2
import math
import mediapipe as mp
import numpy as np
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder

model = tf.keras.models.load_model('model.keras')
encoder = LabelEncoder()
encoder.classes_ = np.load('label_classes.npy', allow_pickle=True)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

def extract_pose_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    body_landmarks_list = []

    if results.pose_landmarks:
        landmarks = []
        for lm in results.pose_landmarks.landmark:
            landmarks.append(lm)

        right_hip_x = landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x
        right_hip_y = landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y
        left_hip_x = landmarks[mp_pose.PoseLandmark.LEFT_HIP].x
        left_hip_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP].y

        center_x = (right_hip_x + left_hip_x) / 2
        center_y = (right_hip_y + left_hip_y) / 2

        max_distance = 0
        for lm in landmarks:
            distance = math.sqrt((lm.x - center_x) ** 2 + (lm.y - center_y) ** 2)
            max_distance = max(max_distance, distance)

        normalized_landmarks = np.array([[(lm.x - center_x) / max_distance,
                                          (lm.y - center_y) / max_distance,
                                          lm.z / max_distance] for lm in landmarks]).flatten()

        body_landmarks_list.append(normalized_landmarks)

    if body_landmarks_list:
        return np.concatenate(body_landmarks_list)
    else:
        return None

cap = cv2.VideoCapture('poses.mp4')

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    image = cv2.flip(image, 1)
    height, width, _ = image.shape

    landmarks = extract_pose_landmarks(image)

    if landmarks is not None:
        landmarks = np.array(landmarks).reshape(1, -1)

        prediction = model.predict(landmarks)
        predicted_class_id = prediction.argmax(axis=1)
        predicted_class = encoder.inverse_transform(predicted_class_id)

        if prediction[0][predicted_class_id] > 0.9:
            cv2.putText(image, f'POSE: {predicted_class[-1].split('_or_')[0]}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(image, "Non clear frame", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    else:
        cv2.putText(image, "No pose !!!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('Yoga Pose Recognition', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
