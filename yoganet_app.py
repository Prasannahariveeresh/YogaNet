# import av
import cv2
import math
import numpy as np
import streamlit as st
import mediapipe as mp
# import plotly.graph_objects as go
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
# import threading
# import queue
import time

# Load model and label encoder
model = load_model('model.keras')
encoder = LabelEncoder()
encoder.classes_ = np.load('label_classes.npy', allow_pickle=True)

# MediaPipe Pose setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

class FEInterface:
    _instance = None

    def __init__(self):
        self.pose = "Pose: Not Clear"
        self.confidence = 0

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(FEInterface, cls).__new__(cls, *args, **kwargs)
        return cls._instance

fe_obj = FEInterface()

# Streamlit UI setup
st.title("YogaNet - Virtual Yoga Tutor 🤖 · 🤸‍♀️")
st.markdown("### Accurate yoga pose classifier which tells how good your pose was")

pose_placeholder = st.empty()
accuracy_placeholder = st.empty()

def extract_pose_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = pose.process(image_rgb)
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        center_x = (right_hip.x + left_hip.x) / 2
        center_y = (right_hip.y + left_hip.y) / 2

        max_distance = max(math.sqrt((lm.x - center_x) ** 2 + (lm.y - center_y) ** 2) for lm in landmarks)
        return np.array([[(lm.x - center_x) / max_distance, (lm.y - center_y) / max_distance, lm.z / max_distance] for lm in landmarks]).flatten()

    return None

def prediction_thread(frame):
    frame = frame.to_ndarray(format="bgr24")
    frame = cv2.flip(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 1)

    landmarks = extract_pose_landmarks(frame)
    if landmarks is not None:
        prediction = model.predict(np.array(landmarks).reshape(1, -1))
        predicted_class_id = prediction.argmax(axis=1)
        predicted_class = encoder.inverse_transform(predicted_class_id)
        confidence = prediction[0][predicted_class_id] * 100

        fe_obj.pose = f"Detected Pose: {predicted_class[0].replace('_', ' ')}" if confidence > 90 else "Pose: Not Clear"
        fe_obj.confidence = confidence[0] if confidence > 90 else 0


webrtc_ctx = webrtc_streamer(
    key="streamer",
    sendback_audio=False,
    video_frame_callback=prediction_thread,
    media_stream_constraints={"video": True, "audio": False}
)

while webrtc_ctx.state.playing:
    pose_placeholder.text(fe_obj.pose)
    accuracy_placeholder.text("Your pose was " + str(fe_obj.confidence) + " % " + "accurate")

    time.sleep(0.1)