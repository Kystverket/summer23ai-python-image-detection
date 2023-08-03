from ultralytics import YOLO
import cv2
import requests
import numpy as np
import os
## For more detailed info
##https://docs.ultralytics.com/modes/predict/#keypoints


class Detector:
    PREDICTION_FOLDER_PATH = "./predictions/"
    def __init__(self,model_path): #pass in path to .pt model 
        self.model = YOLO(model_path)
        self.images_count = 0
        self.clear_output_files() #clear predictions from run of model if it excist

    def find_objects_image( # returns a dictonary containing the classes found and their probability
            self
            ,file #path to image
            ,conf = 0.5 #model confidence level
            ,show_all = False #If true, return probability of all detected instances of objects
            ,save_image = False # if the user want to save the image
            ,save_filename = None
            ):
        image_result = list(self.model.predict(file,conf = conf))[0]
        class_id_list = image_result.boxes.cls.tolist() #find id of all detected classes
        confidence_list = image_result.boxes.conf.tolist() #find confidence of corresponding to the index in the list above
        labels = self.model.names #get class labels
        if save_image:
            im_plot = image_result.plot()
            if save_filename is None:
                save_filename = "image" + str(self.images_count) + ".jpg"
                self.images_count += 1
            print(self.PREDICTION_FOLDER_PATH +  save_filename)
            cv2.imwrite(self.PREDICTION_FOLDER_PATH +  save_filename ,im_plot)
            

            
       
        class_label_list = [labels[class_id] for class_id in class_id_list] #converts class ids to class names
        if show_all: #show all found classes and their probabilities
            class_probabilites = [(class_label_list[index], confidence_list[index]) for index in range(len(confidence_list))]
        else: #only return the distinct classes and their max probability
            max_probability_indexes = [class_label_list.index(unique_element) for unique_element in set(class_label_list)] #find the index of the gretest probability of each class
            class_probabilites = [(class_label_list[max_index], confidence_list[max_index]) for max_index in max_probability_indexes] #make a dictonary with the found classes and their largest probability
        return class_probabilites
    
    def clear_output_files(self):
        
        file_list = os.listdir(self.PREDICTION_FOLDER_PATH)
        for file_name in file_list:
            file_path = os.path.join(self.PREDICTION_FOLDER_PATH, file_name)
            os.remove(file_path)

    
    





    



