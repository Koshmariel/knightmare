# -*- coding: utf-8 -*-
"""
ScoreLivesRecognizer
Input: screenshot
Takes the contents of digits locations, passes them to digit_recognizerm, takes digits and combines them to score and lives.
Output: Score
"""
#%% INITIALIZATION: imports, class

import cv2
import numpy as np
from findFilesInFolder import findFilesInFolder #Recursive function to find all files of an extension type in a folder (and optionally in all subfolders too)
from digit_recognizer import DigitRecognizer

class ScoreLivesRecognizer:
    
    def __init__(self, image_width = 640, image_height = 400, image_channels = 3):
        
        self.image_width = image_width
        self.image_height = image_height
        self.image_channels = image_channels
        self.digit_recognizer = DigitRecognizer()

        
    def get_score (self, screenshot):
        
        width = 16           #digit width & height
        height = 17  
        x_start = 210        #right top corner of right score digit
        y_start = 375
        lives_x_start = 370  #right corner of right lives(rest) digit
        
        

        x = x_start
        y = y_start
        score_imgs = []
        for i in range(1,7):
            score_imgs.append(screenshot[y:y+height, x-width:x])   # X and Y are flipped
            x = x - width
        lives_img = screenshot[y:y+height, lives_x_start-width:lives_x_start]
        return score_imgs, lives_img
    
    def recognize (self, screenshot):
        if len(screenshot.shape) == 3:                                   #check if image is colored
            screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        if not ((screenshot.shape[0] == self.image_height) and (screenshot.shape[1] == self.image_width)):
            screenshot = cv2.resize(screenshot, (self.image_width,self.image_height), interpolation = cv2.INTER_AREA)
        score_imgs, lives_img = self.get_score(screenshot)
        score_imgs.append(lives_img)
        digits = []
        for digit_img in score_imgs:
            digit = self.digit_recognizer.recognize(digit_img)
            digits.append(digit)
        
        #replacing all 10 (category for null digits) with 0
        #all null digits occur before significant figures of the score
        #I made it this way as an experiment, it is faster but for such length it is pampering
        idx = -1
        while True:
            try:
                idx = digits.index(10, idx+1)
            except ValueError:
                break
            else:
                digits[idx] = 0
        lives = digits[6]
        score = digits[0] +         \
                digits[1]*10 +      \
                digits[2]*100 +     \
                digits[3]*1000 +    \
                digits[4]*10000 +   \
                digits[5]*100000
            
        return score, lives

#%% MAIN: debuging

if __name__ == "__main__":
    
    import random
    import matplotlib.pyplot as plt
    import os
    import numpy as np
    


    def plot_figures(figures, nrows = 1, ncols=1):
        """Plot a dictionary of figures.
    
        Parameters
        ----------
        figures : <title, figure> dictionary
        ncols : number of columns of subplots wanted in the display
        nrows : number of rows of subplots wanted in the figure
        """
    
        fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows)
        for ind,title in zip(range(len(figures)), figures):
            axeslist.ravel()[ind].imshow(figures[title], cmap=plt.jet())
            axeslist.ravel()[ind].set_title(title)
            axeslist.ravel()[ind].set_axis_off()
#        plt.tight_layout() # optional





    dir_name = r'G:\Python\KMold\screenshots\categorized\4 gameplay'
    extension = ".jpg"
    
    pathList = []
    pathList = findFilesInFolder(dir_name, pathList, extension, False)
    sample = random.choice(pathList)
    #sample = r"G:\Python\KMold\screenshots\categorized\4 gameplay\2020-07-30-20-21-02.jpg"        #DEBUG
    screenshot = cv2.imread(sample, cv2.IMREAD_GRAYSCALE)
    screenshot = screenshot / 255.
    #screenshot = cv2.imread(sample, cv2.IMREAD_COLOR)                                            #DEBUG
    plt.figure(figsize=(10,10))
    plt.imshow(screenshot)
    plt.show()
    
    score_lives_recognizer = ScoreLivesRecognizer()
    
    #DEBUG
    score_imgs, lives = score_lives_recognizer.get_score(screenshot)
    
    digits_dict = {"x100000": score_imgs[5],
                   "x10000": score_imgs[4],
                   "x1000": score_imgs[3],
                   "x100": score_imgs[2],
                   "x10": score_imgs[1],
                   "x1": score_imgs[0],
                   "Lives": lives
                   }

    plot_figures(digits_dict, ncols=7, nrows=1)
    plt.show()
    
    score, lives = score_lives_recognizer.recognize(screenshot)
    print ('Score =',score,' Lives = ',lives)
    
    
    
