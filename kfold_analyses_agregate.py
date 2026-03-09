import pandas as pd
import numpy as np
from pathlib import Path


def aggregate_kfold_results(project_path):
    project_dir = Path(project_path)
    all_metrics = []

    # 1. Find all results.csv files in the fold directories
    results_files = list(project_dir.glob("runs/pose/runs/pose_kfold/*/results.csv"))

    if not results_files:
        print("No results.csv files found. Ensure training finished correctly.")
        return

    for file in results_files:
        df = pd.read_csv(file)
        # Standard YOLOv8/v11 results.csv columns
        # We take the values from the last epoch (final results)
        last_epoch = df.iloc[-1]

        metrics = {
            "Fold": file.parent.name,
            "Box_mAP50": last_epoch.get("metrics/mAP50(B)", 0),
            "Box_mAP50-95": last_epoch.get("metrics/mAP50-95(B)", 0),
            "Pose_mAP50": last_epoch.get("metrics/mAP50(P)", 0),
            "Pose_mAP50-95": last_epoch.get("metrics/mAP50-95(P)", 0),
            "Fitness": last_epoch.get("fitness", 0)
        }
        all_metrics.append(metrics)

    # 2. Convert to DataFrame for easy math
    summary_df = pd.DataFrame(all_metrics)

    # 3. Calculate Mean and Standard Deviation
    stats = summary_df.drop(columns="Fold").agg(["mean", "std"]).transpose()

    print("\n--- K-Fold Cross-Validation Summary ---")
    print(summary_df.to_string(index=False))
    print("\n--- Final Aggregated Metrics ---")
    print(stats)

    # Save the summary
    summary_df.to_csv(project_dir / "kfold_summary_report.csv", index=False)
    print(f"\nSummary saved to: {project_dir / 'kfold_summary_report.csv'}")


if __name__ == "__main__":
    PROJECT_PATH = "runs/pose_kfold"
    aggregate_kfold_results(PROJECT_PATH)