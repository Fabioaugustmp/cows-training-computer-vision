# Biometric Identification of Dairy Cows via Top-View Keypoint Detection and Machine Learning

**Author:** [Your Name/Project Team]  
**Date:** March 2026  
**Subject:** Visual Computing & Machine Learning Deliverable  

---

## Abstract
This report presents a complete pipeline for the individual identification of 30 dairy cows using non-invasive computer vision. By utilizing top-view 2D imagery, we developed a two-stage system: (1) a Deep Learning Pose Estimation model to detect anatomical keypoints, and (2) a Random Forest Classifier that identifies individual animals based on extracted biometric ratios and skeletal angles. The system achieves high identification accuracy by focusing on growth-invariant morphological proportions.

---

## 1. Introduction
Individual identification in dairy farming is crucial for health monitoring, milk production tracking, and automated management. Traditional methods (tags, RFIDs) can be lost or require proximity. This project implements a visual computing solution based on the research by *JDS (2024)*, leveraging anatomical markers to create a unique biometric profile for each animal.

---

## 2. Methodology

### 2.1 Task 1: Dataset Annotation and Preparation
Data was processed from Label Studio annotations. The raw JSON coordinates were converted into normalized YOLO Pose format.
- **Keypoints defined:** Neck, Withers, Back, Hook (L/R), Hip Ridge, Tail Head, Pin (L/R).
- **Processing:** Implementation in `src/organize_labels.py` ensures spatial normalization and compatibility with the YOLO training engine.

### 2.2 Task 2: Keypoint Detection Model
We employed a **YOLO Pose Estimation** architecture (specifically YOLO26/YOLO11m-pose) to regress the coordinates of 9 anatomical markers.
- **Validation Strategy:** 5-Fold Cross-Validation (`src/kfold_train.py`) was used to ensure the model generalizes across different lighting conditions and cow positions.
- **Outcome:** The model provides the foundation for all subsequent biometric measurements.

### 2.3 Task 3: Biometric Feature Engineering
Identifying animals by size is unreliable due to growth. Therefore, we engineered **growth-invariant features**:
- **Pelvic Ratios:** The ratio between Hook and Pin widths.
- **Body Aspect Ratios:** Body length vs. pelvic width.
- **Spinal Angles:** Calculation of angles at the Withers, Back, and Hip Ridge using vector geometry.
- **Symmetry Indices:** Measuring postural offsets to distinguish individual gait or standing habits.

### 2.4 Task 4: Descriptive Analysis
Using `src/step-4.py`, we conducted a statistical audit of the features.
- **Correlation Analysis:** Identified redundant features to prevent model overfitting.
- **Variance Analysis:** Confirmed that pelvic ratios and spinal angles exhibit high inter-cow variance, making them excellent "biometric fingerprints."

---

## 3. Machine Learning Classification

### 3.1 Task 5: Model Design
A **Random Forest Classifier** was selected for its robustness to tabular data and its ability to rank feature importance.
- **Input:** 12-dimensional biometric feature vector.
- **Output:** Predicted Cow ID (1 of 30).
- **Training:** Performed in `src/step-5.py` with a stratified split to maintain class balance.

### 3.2 Task 6: Evaluation
The final evaluation (`src/step-6.py`) provides a granular view of the system's performance.
- **Metrics:** Precision, Recall, and F1-Score per animal.
- **Key Finding:** The "Pelvic Ratio" and "Spine Prop" were consistently identified as the most discriminating features, confirming the hypothesis that skeletal proportions are unique to individual cows.

---

## 4. System Implementation Summary
The project is organized for high reproducibility:
- **`src/`**: Modular logic for each challenge step.
- **`models/`**: Stores the "Best-in-Class" weights for both Pose and Identification.
- **`results/`**: Transparent logging of accuracy and feature analysis.

## 5. Conclusion
The implemented system successfully fulfills all challenge requirements. By combining Deep Learning for perception (Pose) and traditional Machine Learning for identification (Random Forest), we have created a robust, interpretable, and scalable solution for dairy cow mobility and identity analysis.

---
**References:**
- *Objective dairy cow mobility analysis and scoring system using computer vision–based keypoint detection technique from top-view 2-dimensional videos, JDS, 2024.*
- *Ultralytics YOLO Framework.*
