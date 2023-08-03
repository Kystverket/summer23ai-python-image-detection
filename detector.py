from ultralytics import YOLO
import cv2
import requests
import numpy as np

from io import BytesIO
##https://docs.ultralytics.com/modes/predict/#keypoints


class Detector:
    def __init__(self,model_path): #pass in path to .pt model 
        self.model = YOLO(model_path)

    def find_objects_image( # returns a dictonary containing the classes found and their probability
            self
            ,file #path to image
            ,conf = 0.5 #model confidence level
            ,show_all = False
            ):
        image_result = list(self.model.predict(file,conf = conf))[0]
        class_id_list = image_result.boxes.cls.tolist() #find id of all detected classes
        confidence_list = image_result.boxes.conf.tolist() #find confidence of corresponding to the index in the list above
        labels = self.model.names #get class labels

        class_label_list = [labels[class_id] for class_id in class_id_list] #converts class ids to class names
        if show_all: #show all found classes and their probabilities
            class_probabilites = [(class_label_list[index], confidence_list[index]) for index in range(len(confidence_list))]
        else: #only return the distinct classes and their max probability
            max_probability_indexes = [class_label_list.index(unique_element) for unique_element in set(class_label_list)] #find the index of the gretest probability of each class
            class_probabilites = [(class_label_list[max_index], confidence_list[max_index]) for max_index in max_probability_indexes] #make a dictonary with the found classes and their largest probability
        return class_probabilites
    
    

imo_list = [5181457,9818321,9841782,9857119] 

detector = Detector("models/excavatorV8s.pt")
classified_imo = {}
for imo in imo_list:
    image_url = "https://www.ship-info.com/Vessels/" + str(imo) + ".jpg" # Replace with your image URL
    response = requests.get(image_url)
    image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    #image.save("./loaded/image" + str(imo) + ".jpg")  # Replace with the desired file path
    classified_imo[imo] = detector.find_objects_image(image)

print(classified_imo)



    



