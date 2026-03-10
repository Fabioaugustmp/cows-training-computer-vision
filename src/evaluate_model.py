import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_model():
    print("--- TASK 6: Evaluate Model ---")
    
    # Load artifacts
    try:
        artifacts = joblib.load('../models/cow_id_model.pkl')
    except FileNotFoundError:
        print("Error: '../models/cow_id_model.pkl' not found. Run train_classifier.py first.")
        return

    model = artifacts['model']
    le = artifacts['label_encoder']
    X_test = artifacts['X_test']
    y_test = artifacts['y_test']
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    print(f"Overall Identification Accuracy: {acc:.2%}")
    
    # Feature Importance (Final Confirmation)
    importances = pd.Series(model.feature_importances_, index=X_test.columns).sort_values(ascending=False)
    print("\nMost Discriminating Biometric Features:")
    print(importances)
    
    # Full Report
    print("\nClassification Report per Cow:")
    target_names = [str(c) for c in le.classes_]
    report = classification_report(y_test, y_pred, target_names=target_names)
    print(report)
    
    # Save results
    with open('../results/final_evaluation_report.txt', 'w') as f:
        f.write(f"Accuracy: {acc:.2%}\n\n")
        f.write("Feature Importance:\n")
        f.write(importances.to_string())
        f.write("\n\nClassification Report:\n")
        f.write(report)
    
    print("\nEvaluation complete! Report saved to '../results/final_evaluation_report.txt'")

if __name__ == "__main__":
    evaluate_model()
