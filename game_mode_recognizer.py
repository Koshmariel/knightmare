# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 00:04:43 2020

@author: Kosh
"""

import cv2
from numpy import asarray

class GameModeRecognizer:
    
    """  Class to recognize game mode
    screenshot:       Screenshot from the game
    returns:          Game mode category number
    
    Game modes:
    1: "1 Main menu",
    2: "2 Controls",
    3: "3 Character selection",
    4: "4 Intro",
    5: "5 Level intro",
    6: "6 Gameplay",
    7: "7 Game over",
    8: "8 Continue",
    9: "9 Loading black screen"
    """

    

    
    
    #do not remember why such resolution, but ANN weights were saved for them
    def __init__(self, image_width = 160, image_height = 100, image_channels = 1): 
        from tensorflow.keras.models import load_model
        self.image_width = image_width
        self.image_height = image_height
        self.image_channels = image_channels
        self.classifier=load_model('game_mode_recognizer_model')
        #self.classifier=load_model('game_mode_recognizer_model_simplified')
    

    def recognize(self,screenshot):
        
#        screenshot = asarray(screenshot) #transform image into array of (shape width, height, channels)
        if len(screenshot.shape) == 3:                                   #check if the image is color
            screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2GRAY)
        screenshot = cv2.resize(screenshot, (self.image_width,self.image_height), interpolation = cv2.INTER_AREA)
        ''' DEBUG
        import matplotlib.pyplot as plt
        plt.imshow(screenshot)
        plt.show()
       '''
        #screenshot = screenshot /255      #moved to Environment.make_screenshot
        screenshot = screenshot.reshape((1, screenshot.shape[0], screenshot.shape[1], 1)) #add dimension to transform array into a batch
        classifier_output = self.classifier.predict(screenshot)
        classifier_output = classifier_output[0]                    #decrease dimensions
        category=max(range(len(classifier_output)), key=classifier_output.__getitem__)
        return category + 1
    



if __name__ == "__main__":
    
    import random
#    from keras.preprocessing.image import load_img
    import matplotlib.pyplot as plt
    from numpy import asarray
    from findFilesInFolder import findFilesInFolder
    import time
    
    image_width = 160
    image_height = 100
    image_channels =1

    game_modes = {
                1: "0 Intro",
                2: "1 Game menu",
                3: "2 Level intro",
                4: "3 gameplay",
                5: "4 Credits"
                }    
    
    game_mode_recognizer = GameModeRecognizer()
    
    dir_name = 'G:\Python\KMold\screenshots\categorized'
    extension='.jpg'
    pathList = []
    pathList = findFilesInFolder(dir_name, pathList, extension, True)
    sample = random.choice(pathList)
#    DEBUGGING
#    sample = r'G:\Python\knight-mare\screenshots\2020-07-12-02-40-44.jpg'
#    screenshot = load_img(sample, color_mode="grayscale")
#    screenshot = load_img(sample, color_mode="grayscale", target_size=(image_height,image_width))
    screenshot = cv2.imread(sample, cv2.IMREAD_GRAYSCALE)
    screenshot = screenshot /255.

    
    
    
    start_time=time.time()
    for i in range(100):
        category = game_mode_recognizer.recognize(screenshot)
    end_time=time.time()
    
    plt.imshow(screenshot)
    plt.show()
    
        

    
    print ("Game mode is ", game_modes[category])
    print (f"Recognized in {end_time - start_time} sec")