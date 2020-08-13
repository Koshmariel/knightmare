# -*- coding: utf-8 -*-
"""
Reads screenshots from the specified folder and saves the contents of the digits on score and lives positions
"""
#%% INITIALIZATION
import os
import cv2
import numpy as np
from findFilesInFolder import findFilesInFolder #Recursive function to find all files of an extension type in a folder (and optionally in all subfolders too)

#%% FUNCTIONS

def get_score (screenshot):
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
    lives = screenshot[y:y+height, lives_x_start-width:lives_x_start]
    return score_imgs, lives

   
#%% DIGIT SCRAPPER
    
input_dir_name = os.path.join(os.getcwd(),'screenshots')
output_dir_name = os.path.join(input_dir_name,'digits')
extension = ".jpg"

pathList = []
pathList = findFilesInFolder(input_dir_name, pathList, extension, False)


start_file_num = 1
file_num = start_file_num

for image_path in pathList:
    img = cv2.imread(image_path) 
    score_imgs, lives = get_score(img)
    digit_imgs = score_imgs + [lives]
    
    for img in digit_imgs:
        file_num_str = f'{file_num:05d}'
        file_name = file_num_str + '.jpg'
        file_name_path = os.path.join(output_dir_name, file_name)
        cv2.imwrite(file_name_path, img)
        file_num = file_num + 1

