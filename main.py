import cv2
import mediapipe as mp
import numpy as np
import os
import joblib
from collections import deque

class YogaPoseDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Load model information
        model_info = joblib.load('pose_model_info.joblib')
        self.classifier = model_info['classifier']
        self.scaler = model_info['scaler']
        self.expected_features = model_info['n_features']
        
        # Temporal smoothing
        self.pose_buffer = deque(maxlen=5)
    
    def calculate_angle(self, point1, point2, point3):
        """Calculate angle between three points"""
        x1, y1 = point1
        x2, y2 = point2
        x3, y3 = point3
        
        angle = np.degrees(np.arctan2(y3-y2, x3-x2) - np.arctan2(y1-y2, x1-x2))
        if angle < 0:
            angle += 360
        return angle
    
    def get_pose_angles(self, landmarks):
        """Extract all relevant angles from pose landmarks"""
        points = {}
        for idx, landmark in enumerate(landmarks):
            points[idx] = [landmark.x, landmark.y]
        
        angles = {}
        
        # Left elbow angle
        angles["left_elbow_angle"] = self.calculate_angle(
            points[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value],
            points[self.mp_pose.PoseLandmark.LEFT_ELBOW.value],
            points[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
        )
        
        # Right elbow angle
        angles["right_elbow_angle"] = self.calculate_angle(
            points[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
            points[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value],
            points[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
        )
        
        # Left shoulder angle
        angles["left_shoulder_angle"] = self.calculate_angle(
            points[self.mp_pose.PoseLandmark.LEFT_ELBOW.value],
            points[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value],
            points[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        )
        
        # Right shoulder angle
        angles["right_shoulder_angle"] = self.calculate_angle(
            points[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value],
            points[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
            points[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        )
        
        # Left knee angle
        angles["left_knee_angle"] = self.calculate_angle(
            points[self.mp_pose.PoseLandmark.LEFT_HIP.value],
            points[self.mp_pose.PoseLandmark.LEFT_KNEE.value],
            points[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        )
        
        # Right knee angle
        angles["right_knee_angle"] = self.calculate_angle(
            points[self.mp_pose.PoseLandmark.RIGHT_HIP.value],
            points[self.mp_pose.PoseLandmark.RIGHT_KNEE.value],
            points[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        )
        
        # Hip angles
        angles["left_hip_angle"] = self.calculate_angle(
            points[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value],
            points[self.mp_pose.PoseLandmark.LEFT_HIP.value],
            points[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
        )
        
        angles["right_hip_angle"] = self.calculate_angle(
            points[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
            points[self.mp_pose.PoseLandmark.RIGHT_HIP.value],
            points[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
        )
        
        # Additional angles to match training data
        angles["spine_angle"] = self.calculate_angle(
            points[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value],
            points[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
            points[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        )
        
        angles["neck_angle"] = self.calculate_angle(
            points[self.mp_pose.PoseLandmark.NOSE.value],
            points[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value],
            points[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        )
        
        angles["left_ankle_angle"] = self.calculate_angle(
            points[self.mp_pose.PoseLandmark.LEFT_KNEE.value],
            points[self.mp_pose.PoseLandmark.LEFT_ANKLE.value],
            points[self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value]
        )
        
        angles["right_ankle_angle"] = self.calculate_angle(
            points[self.mp_pose.PoseLandmark.RIGHT_KNEE.value],
            points[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value],
            points[self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value]
        )
        
        angles["left_wrist_angle"] = self.calculate_angle(
            points[self.mp_pose.PoseLandmark.LEFT_ELBOW.value],
            points[self.mp_pose.PoseLandmark.LEFT_WRIST.value],
            points[self.mp_pose.PoseLandmark.LEFT_THUMB.value]
        )
        
        return angles

    def get_pose_features(self, landmarks):
        """Extract features from pose landmarks"""
        features = {}
        
        # Only use first 33 landmarks (skip the last 4)
        for idx in range(33):
            landmark = landmarks[idx]
            features[f"x{idx}"] = landmark.x
            features[f"y{idx}"] = landmark.y
            features[f"z{idx}"] = landmark.z
            features[f"visibility{idx}"] = landmark.visibility
        
        return features

    def detect_pose(self, angles, features):
        """Detect pose using ML model"""
        # Combine angles and features into a single vector
        angle_values = list(angles.values())
        feature_values = list(features.values())
        combined_vector = np.concatenate([angle_values, feature_values])
        
        # Ensure the combined vector has the correct number of features
        if len(combined_vector) != self.expected_features:
            raise ValueError(f"Feature vector length mismatch: expected {self.expected_features}, got {len(combined_vector)}")
        
        # Scale the input
        scaled_vector = self.scaler.transform([combined_vector])
        
        # Get prediction and probability
        prediction = self.classifier.predict(scaled_vector)[0]
        probabilities = self.classifier.predict_proba(scaled_vector)[0]
        confidence = max(probabilities)
        
        # Add to temporal buffer
        self.pose_buffer.append(prediction)
        
        # Get most common pose in buffer
        if len(self.pose_buffer) == self.pose_buffer.maxlen:
            final_prediction = max(set(self.pose_buffer), key=self.pose_buffer.count)
        else:
            final_prediction = prediction
        
        return final_prediction, confidence

    def run(self, video_path):
        cap = cv2.VideoCapture(video_path)
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('output.mp4', fourcc, fps, (frame_width, frame_height))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            if results.pose_landmarks:
                self.mp_draw.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS
                )
                
                angles = self.get_pose_angles(results.pose_landmarks.landmark)
                features = self.get_pose_features(results.pose_landmarks.landmark)
                
                try:
                    detected_pose, confidence = self.detect_pose(angles, features)
                    
                    # Display results
                    cv2.putText(
                        image,
                        f"Pose: {detected_pose} ({confidence:.2f})",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0) if confidence > 0.7 else (0, 255, 255),
                        2
                    )
                except ValueError as e:
                    print(e)
            
            out.write(image)
            cv2.imshow('Yoga Pose Detection', image)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = YogaPoseDetector()
    detector.run("half-moon-pose.mp4")