# Python-image-detection
Template for using Yolo PyTourch models for image detection.


## Documentation `Detector.py`
The `Detector.py` is a simplification of the `YOLO` methods in the `Ultralytics`. It is created to drastically simplify the methods, thus also putting many limitations on the results. For a more detailed analysis, the `YOLO` methods can be used on their own (as shown in XXXX).

- `Detector(model_path)`: Takes in a trained model in the `.pt PyTourch` format. This initializes a detection object which can classify objects from the trained model.
- `find_objects_image(image, conf, show_all, save_image, save_filename)`: Takes in an image and returns a list of found objects with their corresponding probability on the format `(object, prob)`. The method can be modified by adjusting different parameters, with the only required input being the image.
    - `image`: The image where we want to detect objects.
    - `conf = 0.5`: The level of confidence before we add the object to the detected objects.
    - `show_all = False`: When finding multiple instances of the same class. If `True` we show all instances, if `False`, we show the instance with the highest probability.
    - `save_image = False` If `True` we save the image of our prediction with the bounding boxes.
    - `save_filename = None` If `None` we save the images with a default name, else wise we save it with the desired name given.
      
