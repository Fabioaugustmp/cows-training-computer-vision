# Cow Pose Estimation Training

Objective dairy cow mobility analyses and scoring system using computer vision-based keypoint.

## 🚀 How to Train

### 1. Standard Training (Hold-out Validation)
Uses the existing `images/train` and `images/val` folders.
```bash
python train.py
```

### 2. K-Fold Cross-Validation
Use this to verify if the model performance (e.g. mAP 0.99) is consistent across your entire dataset. It splits your data into 5 different parts and trains 5 times.

**Requirements:**
```bash
pip install scikit-learn pyyaml
```

**Run:**
```bash
python kfold_train.py
```
Results will be saved in `runs/pose_kfold/`.

## 🛠️ Project Structure
*   `train.py`: Main training script.
*   `kfold_train.py`: K-Fold validation script for robust performance testing.
*   `prediction.py`: Run inference on new images.
*   `convert.py`: Convert labels from Label Studio to YOLO format.

## 📊 Deciphering the Loss Metrics
During training, you see several "Loss" columns. Think of Loss as the "error rate." We want these as close to zero as possible.

*   **Box_loss:** Error in the bounding box (the rectangle).
*   **Pose_loss:** Error in the 9 keypoint coordinates.
*   **Kobj_loss:** Error in "Keypoint Objectness"—is there actually a point there or not?
*   **Cls_loss:** Classification error (since you only have "cow," this stays low).
*   **DFL & RLE Loss:** These are technical refinements for how the model "smooths" out the box edges and keypoint heatmaps.