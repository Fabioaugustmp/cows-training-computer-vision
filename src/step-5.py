import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def train_classifier():
    print("--- TASK 5: Design and Train ML Model ---")
    df = pd.read_csv('../results/cow_features.csv')
    
    # Preprocessing
    X = df.drop(columns=['filename', 'cow_id'])
    y = df['cow_id']
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Model Selection: Random Forest
    # High robustness to outliers and provides feature importance
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
    model.fit(X_train, y_train)
    
    # Save the model, the label encoder, and the test split for Step 6
    artifacts = {
        'model': model,
        'label_encoder': le,
        'X_test': X_test,
        'y_test': y_test
    }
    joblib.dump(artifacts, '../models/cow_id_model.pkl')
    
    print("Model and training artifacts saved to '../models/cow_id_model.pkl'")

if __name__ == "__main__":
    train_classifier()
