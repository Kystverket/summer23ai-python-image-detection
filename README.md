# Python Image Detection (Kystverket AI summer interns 2023)
Template for using Yolo PyTorch models for image detection.

## Table of Contents
- [About](#about)
- [Getting Started](#getting-started)
- [Using a Pre-Trained Model](#using-a-pre-trained-model)
- [Making, Training, and Using a Custom Data Set](#making-training-and-using-a-custom-data-set)
  - [Making a Custom Dataset](#making-a-custom-dataset)
  - [Training a Model Using a Custom Dataset](#training-a-model-using-a-custom-dataset)
  - [Exporting and Using a Model](#exporting-and-using-a-model)
- [Documentation](#documentation)
  - [Documentation YOLOv8](#documentation-yolov8)
  - [Documentation `Detector.py`](#documentation-detectorpy)

## About
This project aims to simplify the process of image detection in `YOLOv8`, enabling more people to label, train, and use an image detection model.

## Getting Started
1. Clone or download this directory.
2. Navigate to this directory in the command prompt.
3. Run `pip install -r requirements.txt` (this installs all the Python packages required).
4. Try the [`demo.ipynb`](demo.ipynb) notebook to get an understanding of image detection.

## Using a Pre-Trained Model
If you have a pre-trained model (.pt file), you can pass in the relative path to the model when initializing a `YOLO` or a `Detector` object. An example of this is found in the [`demo.ipynb`](demo.ipynb) file.

## Making, Training, and Using a Custom Data Set
### Making a Custom Dataset
Label your dataset by "drawing" boxes around the objects you want to detect and classifying them into categories ("Excavator", "Lighthouse", etc.). Separate the images into training, validation, and test sets. We recommend using [`Roboflow`](https://blog.roboflow.com/getting-started-with-roboflow/). Our own Gravemaskin dataset is available here: [Roboflow - Gravemaskin Dataset](https://universe.roboflow.com/ntnu-3oxpl/gravemaskin).

### Training a Model Using a Custom Dataset
Train the model using a powerful `GPU`. In `Google Colab`, you can take advantage of their built-in `GPU`. A full tutorial is available [here](https://colab.research.google.com/drive/1GLWpHQ8mNH1Mfj1RJzq4046cb_qbuInI?usp=sharing).

### Exporting and Using a Model
After training, you obtain a `PyTorch` file (.pt). You can run your model as with any pre-trained models. See the [tutorial](https://colab.research.google.com/drive/1GLWpHQ8mNH1Mfj1RJzq4046cb_qbuInI?usp=sharing) under "Exporting the model".

## Documentation
### Documentation YOLOv8
- Refer to [the Ultralytics YOLO documentation](https://docs.ultralytics.com/).
- [Object Detection 101](https://colab.research.google.com/drive/1GLWpHQ8mNH1Mfj1RJzq4046cb_qbuInI?usp=sharing)
- [Project Report](https://github.com/Kystverket-AI-KAI/.github/blob/main/profile/Object_Detection_Guide.pdf)

### Documentation `Detector.py`
`Detector.py` simplifies the `YOLO` methods in `Ultralytics`, making them easier to use but with some limitations. For detailed analysis, use the `YOLO` methods directly (as shown in the end of `demo.ipynb`).
- `Detector(model_path)`: Initializes a detection object with a trained model in `.pt PyTorch` format.
- `find_objects_image(image, conf, show_all, save_image, save_filename)`: Detects objects in an image, with customizable parameters.

      
