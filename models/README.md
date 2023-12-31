# Pre-trained models
This directory contains a set of pre-trained models.
## YOLO models
These are pre-trained "standard" YOLO models. The models only differ in speed and precision. With some models being faster with less precision, and others being slower with greater precision. More info can be found on [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/#supported-modes). In increasing order of size:
- [`YOLOv8n.pt`](yolov8n.pt)
- [`YOLOv8s.pt`](yolov8s.pt)
- [`YOLOv8m.pt`](yolov8m.pt)
- [`YOLOv8l.pt`](yolov8l.pt)
- ~`YOLOv8x.pt`~ (Not included due to its size)

## Our custom models
- `ExcavatorV8s.pt`: Model made for detecting Excavators on ships. Trained with the help of the `YOLOv8s.pt`.




