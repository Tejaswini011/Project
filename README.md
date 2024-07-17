# Feature Level Sensor Fusion
* LiDAR Fusion with Vision

## Mid-Level Fusion

* 2D Bboxes from LiDAR are associated with YOLO 2D Bboxes using Greedy Matching Algorithm
* Green Bounding Boxes are detected by YOlO whereas Blue Bounding Boxes are calculated using LiDAR points
* YOLO missed 1 vehicle, whereas 2 vehicles are missed by LiDAR, one of which is half out of frame, at the bottom right side


