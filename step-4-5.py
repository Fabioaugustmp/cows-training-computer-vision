import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

def run_analysis_and_classification():
    # Load the extracted features
    csv_path = 'cow_features.csv'
    if not pd.io.common.file_exists(csv_path):
        print(f"Error: {csv_path} not found. Please run extract_features.py first.")
        return

    df = pd.read_csv(csv_path)
    
    # --- TASK 4: Descriptive Analysis ---
    print("--- TASK 4: Descriptive Analysis ---")
    print(f"Total images processed: {len(df)}")
    print(f"Total unique cows identified: {df['cow_id'].nunique()}")
    
    # Calculate average features per cow to see 'biometric profiles'
    cow_profiles = df.groupby('cow_id').mean(numeric_only=True)
    print("\nSample Cow Biometric Profiles (Means):")
    print(cow_profiles.head())
    
    # Calculate Standard Deviation per feature to see stability
    feature_stability = df.drop(columns=['filename', 'cow_id']).std()
    print("\nFeature Variation (Stability across population):")
    print(feature_stability.sort_values())
    
    # --- TASK 5: Machine Learning Classification ---
    print("\n--- TASK 5 & 6: Classification and Evaluation ---")
    
    # Prepare Features (X) and Labels (y)
    # We exclude filename and the target cow_id
    X = df.drop(columns=['filename', 'cow_id'])
    y = df['cow_id']
    
    # Encode cow_id into integers
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split into Train (80%) and Test (20%) sets
    # Stratify ensures we have samples of every cow in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Train Random Forest Classifier
    # RF is excellent here as it handles non-linear relationships between angles/distances
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Predictions
    y_pred = clf.predict(X_test)
    
    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Identification Accuracy: {accuracy:.2%}")
    
    # Feature Importance - Which biometric marker is most unique?
    importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nMost Important Biometric Identifiers:")
    print(importances)
    
    # Detailed Report
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

if __name__ == "__main__":
    run_analysis_and_classification()
