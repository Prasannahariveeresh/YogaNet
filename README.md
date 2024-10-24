# YogaNet 🧘‍♂️

YogaNet is a real-time yoga pose detection system using a live webcam feed. It is trained on the **Yoga-82** dataset and uses **Mediapipe** for extracting keypoints. With advanced keypoint extraction and pose classification techniques, the model achieves an accuracy of **99%**.

## Features 🚀
- **Real-time yoga pose detection**: Detects and classifies yoga poses from a webcam feed.
- **High accuracy**: Achieved an impressive **99% accuracy** using Mediapipe keypoints.
- **Trained on Yoga-82 dataset**: A comprehensive dataset featuring 82 different yoga poses.
- **Mediapipe integration**: Utilizes Google’s Mediapipe for real-time hand and body keypoint extraction.

## Demo 🎥
![image](https://github.com/user-attachments/assets/bdbc333c-2cf4-4177-ae84-826badf80856)

## Installation 🛠️

### Prerequisites
Ensure you have the following installed:
- **Python 3.8+**
- **OpenCV** for video capture and display
- **Mediapipe** for keypoint extraction
- **TensorFlow**

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/Prasannahariveeresh/YogaNet.git
   cd YogaNet
   ```

2. Install the required dependencies
3. Download the trained model:
   You can use the trained model from the `model.keras`

4. Run the application:
   ```bash
   streamlit run yoganet_app.py  ## For Streamlit app
   python run.py                 ## For detections from poses.mp4
   ```

## Usage 👨‍💻
1. Upon running the script, your webcam feed will appear.
2. The application will detect your body’s keypoints and classify the yoga pose you are performing in real-time.
3. The detected pose will be displayed on the screen along with the webcam feed.

## Dataset 📊
YogaNet is trained on the **Yoga-82** dataset, a large-scale yoga pose dataset featuring **82 different yoga poses** across multiple individuals.

For more details about the dataset, check out the [Yoga-82 Dataset](https://www.kaggle.com/datasets/akashrayhan/yoga-82).

## Model Training 🧠
The model is trained using the following steps:
1. **Mediapipe** is used to extract keypoints from images in the Yoga-82 dataset.
2. The extracted keypoints are then used as input features to the classification model.
3. Achieved **99% accuracy** on test data using a custom architecture built on top of keypoints.
![image](https://github.com/user-attachments/assets/c05d7f96-b061-441d-8328-62177213bc50)

## Contribution 🤝
Feel free to open issues or submit pull requests if you'd like to contribute. Any improvements in pose detection, performance optimizations, or new features are always welcome!
