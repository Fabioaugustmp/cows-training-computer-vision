# Cows Train Project - Visual Computing

This repository contains the source code, data, and models for cow identification using YOLO Pose estimation and biometric feature analysis.

## 📂 Project Structure

- **`src/`**: Python source code for training, feature extraction, and evaluation.
- **`data/`**: Core datasets, labels, and k-fold splits.
- **`config/`**: YAML configuration files for YOLO training.
- **`models/`**: Trained model weights (`.pt`) and identification artifacts (`.pkl`).
- **`results/`**: Extracted features (`.csv`), evaluation reports, and visualization outputs.
- **`docs/`**: Project documentation, guides, and problem descriptions.
- **`runs/`**: (Auto-generated) YOLO training outputs and logs.

## 🚀 Getting Started

### Prerequisites
Install the required dependencies:
```bash
pip install -r requirements.txt
```

### Running the Pipeline
Most scripts are designed to be run from the `src/` directory or from the root using `python src/script_name.py`. 

Example (Training):
```bash
python src/train.py
```

Example (Biometric Extraction):
```bash
python src/extract_features.py
```

## 📊 Documentation
For detailed information on the biometric methodology, please refer to the files in the `docs/` folder.
