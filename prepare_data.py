import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_prepare_data():
    dataset_dir = 'dataset/Yoga_Poses-Dataset-main/Yoga_Poses-Dataset-main/Results/'
    processed_data = []
    labels = []
    
    for filename in os.listdir(dataset_dir):
        if filename.endswith('_Angles.csv'):
            pose_name = filename.replace('Dataset_', '').replace('_Angles.csv', '')
            print(f"Processing {pose_name}...")
            
            # Load angle data
            angle_csv_path = os.path.join(dataset_dir, filename)
            angle_df = pd.read_csv(angle_csv_path)
            angle_columns = angle_df.columns[2:]  # Skip first two columns
            angle_data = angle_df[angle_columns].values
            
            # Load feature data
            feature_csv_path = os.path.join(dataset_dir, f'Dataset_{pose_name}.csv')
            feature_df = pd.read_csv(feature_csv_path)
            feature_columns = feature_df.columns[2:]  # Skip first two columns
            feature_data = feature_df[feature_columns].values
            
            # Print feature information for the first file
            if len(processed_data) == 0:
                print(f"Number of angle features: {len(angle_columns)}")
                print(f"Angle features: {angle_columns.tolist()}")
                print(f"Number of pose features: {len(feature_columns)}")
                print(f"Pose features: {feature_columns.tolist()}")
            
            # Combine data for each frame
            for i in range(len(angle_data)):
                combined_vector = np.concatenate([angle_data[i], feature_data[i]])
                processed_data.append(combined_vector)
                labels.append(pose_name)
    
    return np.array(processed_data), np.array(labels)

if __name__ == "__main__":
    X, y = load_and_prepare_data()
    np.save('processed_data.npy', X)
    np.save('labels.npy', y)
    print(f"Saved {len(X)} samples with {X.shape[1]} features each")
    
    # Save feature information for reference
    feature_info = {
        'n_features': X.shape[1],
        'sample_shape': X.shape
    }
    np.save('feature_info.npy', feature_info)