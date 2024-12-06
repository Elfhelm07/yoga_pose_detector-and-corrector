# Yoga Pose Detector and Corrector

A computer vision-based system that can detect various yoga poses and provide real-time feedback for pose correction, with a specific focus on the Warrior II pose. The project uses MediaPipe for pose estimation and machine learning for pose classification and correction guidance.

## 🎯 Features

- **Pose Detection**: Identifies multiple yoga poses using machine learning
- **Real-time Feedback**: Provides instant feedback on pose alignment
- **Pose Correction**: Offers visual guidance for proper pose alignment (currently implemented for Warrior II)
- **Angle Analysis**: Calculates key joint angles for precise pose evaluation
- **Visual Overlay**: Shows ideal pose alignment for comparison
- **Temporal Smoothing**: Implements pose prediction smoothing to reduce jitter

## 🛠️ Technologies Used

- **Python 3.x**
- **OpenCV** - For video processing and visualization
- **MediaPipe** - For pose landmark detection
- **NumPy** - For numerical computations
- **Scikit-learn** - For pose classification
- **Joblib** - For model serialization

## 📊 System Architecture

### 1. Pose Detection System (`main.py`)

- Uses MediaPipe Pose for landmark detection
- Extracts 33 key points with their x, y, z coordinates and visibility scores
- Calculates various angles between joints
- Implements ML-based pose classification
- Features temporal smoothing using a rolling window

### 2. Pose Correction System (`static_warrior_pose.py`)

- Focuses on Warrior II pose correction
- Provides real-time angle measurements
- Offers visual guidance with ideal pose overlay
- Generates specific correction instructions
- Implements pose scaling to match user dimensions

## 🎬 Demo Videos

### Pose Detection

The `pose_detection.mp4` demonstrates the system's ability to:

- Detect and classify different yoga poses
- Display confidence scores
- Show real-time pose landmarks

### Pose Correction

The `pose_correction.mp4` shows:

- Warrior II pose analysis
- Real-time feedback and corrections
- Ideal pose overlay
- Angle measurements and adjustments

## 📐 Key Measurements

The system analyzes several key angles including:

- Knee angles (left and right)
- Elbow angles
- Shoulder angles
- Hip angles
- Spine angle
- Neck angle
- Ankle angles
- Wrist angles

## 💻 Installation

1. Clone the repository:

2. Install required packages:

```bash
pip install opencv-python mediapipe numpy scikit-learn joblib
```

## 🚀 Usage

### For Pose Detection:

```bash
python main.py
```

### For Warrior II Pose Correction:

```bash
python static_warrior_pose.py
```

## 📝 Implementation Details

### Pose Detection

- Implements feature extraction from pose landmarks
- Uses scaled coordinates and visibility scores
- Applies temporal smoothing with a 5-frame buffer
- Provides confidence scores for pose classification

### Pose Correction

- Defines ideal angles for Warrior II pose
- Implements tolerance ranges for each angle
- Provides specific correction instructions
- Scales ideal pose overlay to match user's dimensions
- Offers visual feedback through semi-transparent overlays
