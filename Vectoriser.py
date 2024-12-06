import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def generate_pose_vectors():
    dataset_dir = 'dataset/Yoga_Poses-Dataset-main/Yoga_Poses-Dataset-main/Results/'
    poses_dir = 'poses/'
    os.makedirs(poses_dir, exist_ok=True)
    
    # Initialize scalers
    angle_scaler = StandardScaler()
    feature_scaler = StandardScaler()
    
    for filename in os.listdir(dataset_dir):
        if filename.endswith('_Angles.csv'):
            pose_name = filename.replace('Dataset_', '').replace('_Angles.csv', '')
            print(f"Processing {pose_name}...")
            
            # Load and normalize angle data
            angle_csv_path = os.path.join(dataset_dir, filename)
            angle_df = pd.read_csv(angle_csv_path)
            angle_columns = angle_df.columns[2:]  # Skip first two columns
            angle_data = angle_df[angle_columns].values
            
            # Fit and transform angle data
            normalized_angles = angle_scaler.fit_transform(angle_data)
            angle_vector = {
                'means': angle_scaler.mean_,
                'scales': angle_scaler.scale_,
                'column_names': list(angle_columns)
            }
            
            # Load and normalize feature data
            feature_csv_path = os.path.join(dataset_dir, f'Dataset_{pose_name}.csv')
            feature_df = pd.read_csv(feature_csv_path)
            feature_columns = feature_df.columns[2:]  # Skip first two columns
            feature_data = feature_df[feature_columns].values
            
            # Fit and transform feature data
            normalized_features = feature_scaler.fit_transform(feature_data)
            feature_vector = {
                'means': feature_scaler.mean_,
                'scales': feature_scaler.scale_,
                'column_names': list(feature_columns)
            }
            
            # Combine vectors with metadata
            pose_vector = {
                'angles': angle_vector,
                'features': feature_vector,
                'pose_name': pose_name
            }
            
            # Save vector
            vector_path = os.path.join(poses_dir, f'{pose_name}_vector.npy')
            np.save(vector_path, pose_vector)
            print(f"Saved vector for {pose_name}")
    
    print("Vectors regenerated successfully!")

if __name__ == "__main__":
    generate_pose_vectors()