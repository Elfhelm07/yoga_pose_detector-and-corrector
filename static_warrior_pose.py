import cv2
import mediapipe as mp
import numpy as np
from math import atan2, degrees

class YogaPoseDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Load pose definitions
        self.pose_definitions = self.load_pose_definitions()
        
    def load_pose_definitions(self):
        # Example pose definition for Warrior II
        return {
            "warrior2": {
                "angles": {
                    "right_knee": {"ideal": 90, "tolerance": 15},
                    "left_knee": {"ideal": 180, "tolerance": 15},
                    "arms": {"ideal": 180, "tolerance": 15}
                }
            }
        }
    
    def calculate_angle(self, point1, point2, point3):
        """Calculate angle between three points"""
        x1, y1 = point1
        x2, y2 = point2
        x3, y3 = point3
        
        angle = degrees(atan2(y3-y2, x3-x2) - atan2(y1-y2, x1-x2))
        if angle < 0:
            angle += 360
        return angle

    def get_pose_angles(self, landmarks):
        """Extract relevant angles from pose landmarks"""
        points = []
        for landmark in landmarks:
            points.append([landmark.x, landmark.y])
        points = np.array(points)
        
        right_knee = self.calculate_angle(
            points[self.mp_pose.PoseLandmark.RIGHT_HIP.value],
            points[self.mp_pose.PoseLandmark.RIGHT_KNEE.value],
            points[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        )
        
        left_knee = self.calculate_angle(
            points[self.mp_pose.PoseLandmark.LEFT_HIP.value],
            points[self.mp_pose.PoseLandmark.LEFT_KNEE.value],
            points[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        )
        
        arms = self.calculate_angle(
            points[self.mp_pose.PoseLandmark.RIGHT_WRIST.value],
            points[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
            points[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
        )
        
        return {
            "right_knee": right_knee,
            "left_knee": left_knee,
            "arms": arms
        }

    def check_pose(self, angles, pose_name="warrior2"):
        """Check if current pose matches the target pose"""
        pose_def = self.pose_definitions[pose_name]
        corrections = []
        all_correct = True
        
        for angle_name, angle_value in angles.items():
            ideal = pose_def["angles"][angle_name]["ideal"]
            tolerance = pose_def["angles"][angle_name]["tolerance"]
            
            if abs(angle_value - ideal) > tolerance:
                all_correct = False
                if angle_value < ideal:
                    corrections.append(f"Increase {angle_name.replace('_', ' ')} angle")
                else:
                    corrections.append(f"Decrease {angle_name.replace('_', ' ')} angle")
        
        return all_correct, corrections

    def draw_correct_pose(self, image, landmarks, color=(0, 255, 0)):
        """Draw the correct pose landmarks and connections"""
        # Draw points with larger radius and semi-transparency
        overlay = image.copy()
        for point in landmarks.values():
            cv2.circle(overlay, point, 8, color, -1)
        
        # Define essential connections for warrior pose
        essential_connections = [
            # Legs
            (self.mp_pose.PoseLandmark.LEFT_HIP.value, self.mp_pose.PoseLandmark.LEFT_KNEE.value),
            (self.mp_pose.PoseLandmark.LEFT_KNEE.value, self.mp_pose.PoseLandmark.LEFT_ANKLE.value),
            (self.mp_pose.PoseLandmark.RIGHT_HIP.value, self.mp_pose.PoseLandmark.RIGHT_KNEE.value),
            (self.mp_pose.PoseLandmark.RIGHT_KNEE.value, self.mp_pose.PoseLandmark.RIGHT_ANKLE.value),
            # Torso
            (self.mp_pose.PoseLandmark.LEFT_SHOULDER.value, self.mp_pose.PoseLandmark.LEFT_HIP.value),
            (self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value, self.mp_pose.PoseLandmark.RIGHT_HIP.value),
            (self.mp_pose.PoseLandmark.LEFT_SHOULDER.value, self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value),
            # Arms
            (self.mp_pose.PoseLandmark.LEFT_SHOULDER.value, self.mp_pose.PoseLandmark.LEFT_ELBOW.value),
            (self.mp_pose.PoseLandmark.LEFT_ELBOW.value, self.mp_pose.PoseLandmark.LEFT_WRIST.value),
            (self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value, self.mp_pose.PoseLandmark.RIGHT_ELBOW.value),
            (self.mp_pose.PoseLandmark.RIGHT_ELBOW.value, self.mp_pose.PoseLandmark.RIGHT_WRIST.value),
        ]
        
        # Draw the connections with semi-transparency
        for connection in essential_connections:
            if connection[0] in landmarks and connection[1] in landmarks:
                start_point = landmarks[connection[0]]
                end_point = landmarks[connection[1]]
                cv2.line(overlay, start_point, end_point, color, 3)
        
        # Add the overlay with transparency
        alpha = 0.6  # Transparency factor
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    def calculate_distance(self, point1, point2):
        """Calculate the Euclidean distance between two points"""
        return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def scale_ideal_pose(self, detected_landmarks, ideal_landmarks):
        """Scale and position the ideal pose based on detected pose"""
        # Get the center point (hip center) of detected pose
        detected_hip_center = (
            (detected_landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value][0] + 
             detected_landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value][0]) // 2,
            (detected_landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value][1] + 
             detected_landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value][1]) // 2
        )
        
        # Calculate the user's full height from top of head to ankle
        user_height = self.calculate_distance(
            detected_landmarks[self.mp_pose.PoseLandmark.NOSE.value],
            detected_landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        )
        
        # Scale multiplier to match user dimensions
        SCALE_FACTOR = 1.5  # Adjust this value to make the overlay larger or smaller
        
        # Calculate relative positions from hip center
        scaled_landmarks = {}
        for idx, (x, y) in ideal_landmarks.items():
            # Convert from relative (0-1) to pixel coordinates
            rel_x = x - 0.5  # Convert from 0-1 scale to -0.5 to 0.5 scale
            rel_y = y - 0.5
            
            # Scale based on user's full height with multiplier
            scaled_x = rel_x * user_height * SCALE_FACTOR
            scaled_y = rel_y * user_height * SCALE_FACTOR
            
            # Add to hip center position
            final_x = int(detected_hip_center[0] + scaled_x)
            final_y = int(detected_hip_center[1] + scaled_y)
            
            scaled_landmarks[idx] = (final_x, final_y)
        
        return scaled_landmarks

    def run(self, video_path):
        cap = cv2.VideoCapture(video_path)
        
        # Define ideal pose landmarks for Warrior II (expanded set)
        ideal_landmarks = {
            # Legs
            self.mp_pose.PoseLandmark.LEFT_HIP.value: (0.4, 0.5),
            self.mp_pose.PoseLandmark.LEFT_KNEE.value: (0.3, 0.7),
            self.mp_pose.PoseLandmark.LEFT_ANKLE.value: (0.3, 0.9),
            self.mp_pose.PoseLandmark.RIGHT_HIP.value: (0.6, 0.5),
            self.mp_pose.PoseLandmark.RIGHT_KNEE.value: (0.7, 0.7),
            self.mp_pose.PoseLandmark.RIGHT_ANKLE.value: (0.7, 0.9),
            # Upper body
            self.mp_pose.PoseLandmark.LEFT_SHOULDER.value: (0.4, 0.3),
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value: (0.6, 0.3),
            self.mp_pose.PoseLandmark.LEFT_ELBOW.value: (0.2, 0.3),
            self.mp_pose.PoseLandmark.RIGHT_ELBOW.value: (0.8, 0.3),
            self.mp_pose.PoseLandmark.LEFT_WRIST.value: (0.1, 0.3),
            self.mp_pose.PoseLandmark.RIGHT_WRIST.value: (0.9, 0.3),
        }
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
        out = cv2.VideoWriter('output.mp4', fourcc, fps, (frame_width, frame_height))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            height, width = frame.shape[:2]
            font_scale = min(width, height) * 0.001
            font_thickness = max(1, int(font_scale * 2))
            padding = int(height * 0.05)
                
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
                is_correct, corrections = self.check_pose(angles)
                
                if is_correct:
                    cv2.putText(image, "Perfect Pose!", 
                              (padding, padding),
                              cv2.FONT_HERSHEY_SIMPLEX, 
                              font_scale, (0, 255, 0), 
                              font_thickness)
                else:
                    y = padding
                    for correction in corrections:
                        text_size = cv2.getTextSize(correction, 
                                                  cv2.FONT_HERSHEY_SIMPLEX,
                                                  font_scale, 
                                                  font_thickness)[0]
                        x = min(padding, width - text_size[0] - padding)
                        cv2.putText(image, correction, 
                                  (x, y),
                                  cv2.FONT_HERSHEY_SIMPLEX, 
                                  font_scale, (0, 0, 255), 
                                  font_thickness)
                        y += int(padding * 1.5)
                
                y = padding * 3
                debug_font_scale = font_scale * 0.7
                for angle_name, angle_value in angles.items():
                    text = f"{angle_name}: {angle_value:.1f}"
                    text_size = cv2.getTextSize(text, 
                                              cv2.FONT_HERSHEY_SIMPLEX,
                                              debug_font_scale, 
                                              font_thickness)[0]
                    x = min(padding, width - text_size[0] - padding)
                    cv2.putText(image, text,
                              (x, y),
                              cv2.FONT_HERSHEY_SIMPLEX, 
                              debug_font_scale, 
                              (255, 255, 255), 
                              max(1, font_thickness - 1))
                    y += int(padding * 0.8)
                
                # Scale the ideal pose based on detected pose
                detected_landmarks = {
                    idx: (int(landmark.x * width), int(landmark.y * height))
                    for idx, landmark in enumerate(results.pose_landmarks.landmark)
                }
                scaled_ideal_landmarks = self.scale_ideal_pose(detected_landmarks, ideal_landmarks)
                
                # Draw the correct pose overlay
                self.draw_correct_pose(image, scaled_ideal_landmarks, color=(255, 0, 0))
            
            # Write the frame to the output video
            out.write(image)
            
            cv2.imshow('Yoga Pose Detection', image)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = YogaPoseDetector()
    detector.run("warrior_pose2.mp4")