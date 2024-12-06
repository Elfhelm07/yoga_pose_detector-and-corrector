import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib

def train_pose_classifier():
    # Load processed data
    X = np.load('processed_data.npy')
    y = np.load('labels.npy')
    feature_info = np.load('feature_info.npy', allow_pickle=True).item()
    
    print(f"Training with {feature_info['n_features']} features")
    print(f"Data shape: {feature_info['sample_shape']}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train SVM classifier
    svm_classifier = SVC(kernel='rbf', probability=True)
    svm_classifier.fit(X_train_scaled, y_train)
    
    # Save model, scaler, and feature information
    model_info = {
        'classifier': svm_classifier,
        'scaler': scaler,
        'n_features': feature_info['n_features'],
        'feature_shape': feature_info['sample_shape']
    }
    joblib.dump(model_info, 'pose_model_info.joblib')
    
    # Print performance metrics
    train_score = svm_classifier.score(X_train_scaled, y_train)
    test_score = svm_classifier.score(X_test_scaled, y_test)
    print(f"Training accuracy: {train_score:.2f}")
    print(f"Testing accuracy: {test_score:.2f}")

if __name__ == "__main__":
    train_pose_classifier()