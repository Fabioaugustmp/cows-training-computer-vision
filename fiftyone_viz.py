import os
import pandas as pd
import fiftyone as fo
from ultralytics import YOLO
from pathlib import Path
import numpy as np

# --- Configuration ---
MODEL_PATH = 'runs/pose/runs/pose_kfold/cow_pose_kfold_1/weights/best.pt'
CSV_PATH = 'cow_features.csv'
IMAGES_DIR = 'dataset/images'


def create_viz_dataset():
    # 1. Load the feature data
    if not os.path.exists(CSV_PATH):
        print(f"Error: {CSV_PATH} not found.")
        return
    df = pd.read_csv(CSV_PATH)

    # 2. Create FiftyOne Dataset
    dataset = fo.Dataset(name="cow_biometrics_analysis", overwrite=True)

    # 3. Load the YOLO model for visual overlay
    print("Loading YOLO model for keypoint visualization...")
    model = YOLO(MODEL_PATH)

    # 4. Add samples to dataset
    samples = []
    print("Building dataset samples...")

    image_paths = list(Path(IMAGES_DIR).rglob("*.jpg"))

    for img_path in image_paths:
        filename = img_path.name

        # Get matching features from CSV
        feature_row = df[df['filename'] == filename]
        if feature_row.empty:
            continue

        row = feature_row.iloc[0]

        # Create Sample
        sample = fo.Sample(filepath=str(img_path.absolute()))

        # CORREÇÃO: Converter explicitamente para float/int nativos do Python
        # Isso evita o erro bson.errors.InvalidDocument
        sample["cow_id"] = fo.Classification(label=str(row["cow_id"]))
        sample["pelvic_ratio"] = float(row["pelvic_ratio"])
        sample["body_aspect"] = float(row["body_aspect"])
        sample["angle_withers"] = float(row["angle_withers"])
        sample["symmetry_idx"] = float(row["symmetry_idx"])

        # 5. Predict Keypoints for visualization
        results = model.predict(str(img_path), conf=0.6, verbose=False)
        for r in results:
            if r.keypoints is not None and len(r.keypoints.data) > 0:
                # Convert YOLO keypoints to FiftyOne format
                # Garantimos que os dados saiam da GPU/Tensor para CPU e viragem lista Python
                kpts = r.keypoints.data[0].cpu().numpy()

                w, h = r.orig_shape[1], r.orig_shape[0]

                # Criar lista de pontos normalizados [0, 1]
                points = []
                for x, y, conf in kpts:
                    points.append([float(x / w), float(y / h)])

                # Adicionar ao sample como Keypoint do FiftyOne
                sample["pose_predictions"] = fo.Keypoints(
                    keypoints=[fo.Keypoint(points=points, label="cow_skeleton")]
                )

        samples.append(sample)

    # 6. Adicionar amostras ao dataset
    dataset.add_samples(samples)

    print("\nDataset ready! Launching FiftyOne...")
    print("Dica de Arquiteto: No App, use o menu 'Scalars' para identificar por que a acurácia está em 25%.")

    session = fo.launch_app(dataset)
    session.wait()


if __name__ == "__main__":
    create_viz_dataset()