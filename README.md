1 - train.py
2 - prediction.py
3 - conversion.py

2. Deciphering the Loss Metrics
During training, you see several "Loss" columns. Think of Loss as the "error rate." We want these as close to zero as possible.

Box_loss: Error in the bounding box (the rectangle).

Pose_loss: Error in the 9 keypoint coordinates.

Kobj_loss: Error in "Keypoint Objectness"—is there actually a point there or not?

Cls_loss: Classification error (since you only have "cow," this stays low).

DFL & RLE Loss: These are technical refinements for how the model "smooths" out the box edges and keypoint heatmaps.