# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 00:04:43 2020

@author: Kosh
"""

import cv2
from numpy import asarray

class DigitRecognizer:
    
    from tensorflow.keras.models import load_model
    from numpy import asarray
    #classifier=load_model('digit_recognition_model')
    classifier=load_model('digit_recognition_model_simplified')
    
    def __init__(self, image_width = 16, image_height = 17, image_channels = 1):
        
        self.image_width = image_width
        self.image_height = image_height
        self.image_channels = image_channels

    

    def recognize(self,screenshot):

# color check moved to score_lives_recognizer, ScoreLivesRecognizer.recognize() would pass grayscale
#        if len(screenshot.shape) == 3:                                   #check if image is colored
#            screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

# ScoreLivesRecognizer.recognize() would pass right size
#        if not ((screenshot.shape[0] == self.image_height) and (screenshot.shape[1] == self.image_width)):
#           screenshot = cv2.resize(screenshot, (self.image_width,self.image_height), interpolation = cv2.INTER_AREA)
# RENAME screenshot to DIGIT, it is a croped digit image
#        screenshot = asarray(screenshot) #transform image into array of (shape width, height, channels)    #moved to Environment.make_scrrenshot
#        screenshot = screenshot /255                                                                       #moved to Environment.make_scrrenshot
        screenshot = screenshot.reshape((1, screenshot.shape[0], screenshot.shape[1], 1)) #add dimension to transform array into a batch
        classifier_output = self.classifier.predict(screenshot)
        classifier_output = classifier_output[0]                    #decrease dimensions
        category=max(range(len(classifier_output)), key=classifier_output.__getitem__)

#        print("ANN output = {}".format(classifier_output))          #DEBUG
#        print("Category = ",category)                               #DEBUG
        return category
    


if __name__ == "__main__":
    
    import random
#    from keras.preprocessing.image import load_img
    import matplotlib.pyplot as plt
    from findFilesInFolder import findFilesInFolder
#    from numpy import asarray
    
    
    image_width = 16
    image_height = 17
    image_channels =1
    
    digits = {
                1: "0",
                2: "1",
                3: "2",
                4: "3",
                5: "4",
                6: "5",
                7: "6",
                8: "7",
                9: "8",
                10: "9",
                }
    
    digit_recognizer = DigitRecognizer()
    
    dir_name = 'G:\Python\KMold\screenshots\digits'
    extension='.jpg'
    pathList = []
    pathList = findFilesInFolder(dir_name, pathList, extension, True)
    sample = random.choice(pathList)
#    sample = r"G:\Python\knight-mare\screenshots\digits\None\00102.jpg"   #DEBUG
    screenshot = cv2.imread(sample, cv2.IMREAD_GRAYSCALE)
#    screenshot = cv2.imread(sample, cv2.IMREAD_COLOR)                     #DEBUG
    
    

    category = digit_recognizer.recognize(screenshot)
    
    plt.imshow(screenshot)
    plt.show()
    

   
    
    print ("Digit is ",digits[category + 1])
    
    import time
    n = 1000
    start_time=time.time()
    for i in range(n):
        sample = random.choice(pathList)
        screenshot = cv2.imread(sample, cv2.IMREAD_GRAYSCALE)
        category = digit_recognizer.recognize(screenshot)

    end_time=time.time()
    
    print (f"{n} digits recognized in {end_time - start_time} sec")