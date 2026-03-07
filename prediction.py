from ultralytics import YOLO
import cv2

def predict():
    # 1. Load your newly trained custom model
    # Make sure the path matches your 'Results saved to' location from the last run
    model = YOLO(r'C:\Workspace\cows-train\runs\pose\train\weights\best.pt')

    # 2. Run prediction on a new image
    # Replace 'test_cow.jpg' with a path to a cow image the model hasn't seen
    results = model.predict(
        source='RLC1_00_20260102064248_baia12_RLC1.jpg',
        save=True,
        conf=0.25,
        imgsz = 640,  # Force standard size
        augment = True  # Enable Test-Time Augmentation (TTA) to help with the weird angle
    )

    # 3. Look at the results
    for r in results:
        # Print the 9 keypoints (x, y coordinates) for the first cow detected
        if r.keypoints is not None:
            print("Detected Keypoints (x, y):")
            print(r.keypoints.xy)

            # The annotated image will be saved in 'runs/pose/predict'
    print(f"Check your results folder for the visual output!")


# The 'main' idiom is required for multiprocessing to work safely
if __name__ == '__main__':
    # freeze_support()  # Only needed if you're turning this into an .exe
    predict()