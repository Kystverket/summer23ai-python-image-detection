# Python Image Detection (Kystverket AI summer interns 2023)
Template for using Yolo PyTorch models for image detection. 

## Table of Contents
- [About](#about)
- [Getting Started](#getting-started)
- [Using a Pre-Trained Model](#using-a-pre-trained-model)
- [Making, Training, and Using a Custom Data Set](#making-training-and-using-a-custom-data-set)
  - [Making a Custom Dataset](#making-a-custom-dataset)
  - [Training a Model Using a Custom Dataset](#training-a-model-using-a-custom-dataset)
  - [Using a Model](#using-a-model)
- [Documentation](#documentation)
  - [Documentation `Detector.py`](#documentation-detectorpy)

## About
This project aims to simplify the process of image detection in `YOLOv8`, in order to enable more people to label, train and use an image detection model.
## Getting started
1. Clone or download this directory
2. In the command prompt `cd` into the folder
3. Run `pip install -r requirements.txt` (this installs all the Python packages required)
4. Try the [`demo.ipynb`](vg.no) notebook to get an understanding of image detection.

## Using a pre-trained model
If you have a pre-trained model (.pt file), you can just pass the in the relative path to the model when initializing a `YOLO` or a `Detector` object. An example of this is found in the `demo.ipynb` file.

## Making, training, and using a custom data set
When wanting to detect custom, we need to make our own model. In short, this is done by labeling a set of images and running training a model using our labeled images.

### Making a custom dataset
When we label our dataset, we "draw" boxes around the objects we want to detect and classify them to their category ("Excavator", "Lighthouse" etc). We also need to separate the images into a training, validation, and test set. There are many ways of doing this, but we recommend using [`Roboflow`](https://blog.roboflow.com/getting-started-with-roboflow/). They provide a simple user interface for labeling images and a quick way of exporting the dataset.
### Training a model using a custom dataset
When training the model, we need a powerfull `GPU`. In `Google Colab` we can take advantage of their built in `GPU` to train our model. A full tutorial can be found [here](https://colab.research.google.com/drive/1GLWpHQ8mNH1Mfj1RJzq4046cb_qbuInI?usp=sharing).
### Exporting and using a model 
After training the model, you obtain a `PyTorch` file (.pt). Using this file, you can run your model as with any of the pre-trained models. See the [tutorial](https://colab.research.google.com/drive/1GLWpHQ8mNH1Mfj1RJzq4046cb_qbuInI?usp=sharing) under "Exporting the model".


## Documentation
### Documentation `Detector.py`
The `Detector.py` is a simplification of the `YOLO` methods in the `Ultralytics`. It is created to drastically simplify the methods, thus also putting many limitations on the results. For a more detailed analysis, the `YOLO` methods can be used on their own (as shown in XXXX).

- `Detector(model_path)`: Takes in a trained model in the `.pt PyTorch` format. This initializes a detection object which can classify objects from the trained model.
- `find_objects_image(image, conf, show_all, save_image, save_filename)`: Takes in an image and returns a list of found objects with their corresponding probability on the format `(object, prob)`. The method can be modified by adjusting different parameters, with the only required input being the image.
    - `image`: The image where we want to detect objects.
    - `conf = 0.5`: The level of confidence before we add the object to the detected objects.
    - `show_all = False`: When finding multiple instances of the same class. If `True` we show all instances, if `False`, we show the instance with the highest probability.
    - `save_image = False` If `True` we save the image of our prediction with the bounding boxes.
    - `save_filename = None` If `None` we save the images with a default name, else wise we save it with the desired name given.
      
