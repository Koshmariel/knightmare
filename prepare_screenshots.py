# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 15:02:12 2020

@author: Kosh
"""
#%% INITIALIZATION

import numpy as np
#from PIL import ImageGrab
import cv2
import os
import time
import pyautogui
import win32gui
import subprocess


#%% FUNCTIONS

#Usefull to get a list of window names
def winEnumHandler( hwnd, ctx ):
    if win32gui.IsWindowVisible( hwnd ):
        print (hex(hwnd), win32gui.GetWindowText( hwnd ))

#win32gui.EnumWindows( winEnumHandler, None )


def rungame():
    """
    Runs the game and moves it's window to the top left corner'
    

    Returns
    -------
    None.

    """


    #set a working directory to the game folder and then restore it
    #work_dir = os.getcwd()
    #km_dir=r'G:\Python\knight-mare\kmare_gold_win32'
    #os.chdir(km_dir)
    #os.startfile(r'G:\Python\knight-mare\kmare_gold_win32\Kmare.exe')
    #os.chdir(work_dir)
    
    #subprocess is modern way of running process
    #subprocess.run waits for process to complete
    subprocess.Popen(r'G:\Python\KMold\Game\DOSBox-0.74-2\DOSBox.exe', cwd=r'G:\Python\KMold\Game\DOSBox-0.74-2')
    
    time.sleep(2)  #wait until a game will start
    
    hwnd = win32gui.FindWindow(None, "DOSBox 0.74-2, Cpu speed:     3000 cycles, Frameskip  0, Program:     MYTH")
    rect = win32gui.GetWindowRect(hwnd) #get window size
    width = rect[2] - rect[0]
    height = rect[3] - rect[1]
    #                   Win has an invisible border of 7 pixels, but it looks like KM windows actual size is a bit larger
    win32gui.MoveWindow(hwnd, -4, 0, width, height, True) #move the window to the top left corner
    
    time.sleep(8)  #wait until a game loads to the main menu

def make_screenshot():

    window_title='DOSBox 0.74-2, Cpu speed:     3000 cycles, Frameskip  0, Program:     MYTH'
    hwnd = win32gui.FindWindow(None, window_title)
    if hwnd:
        #win32gui.SetForegroundWindow(hwnd)
        x, y, x1, y1 = win32gui.GetClientRect(hwnd)
        x, y = win32gui.ClientToScreen(hwnd, (x, y))
        x1, y1 = win32gui.ClientToScreen(hwnd, (x1 - x, y1 - y))
        screenshot = pyautogui.screenshot(region=(x, y, x1, y1))
        
        screenshot = np.array(screenshot)
        return screenshot
    else:
        print('Window not found!')
    






def img_resize (img, scale_percent):
    """  Function to resize an image to specified percent of original size preserving aspect ratio 
    img:              Original image
    scale_percent:    Specified percent of original size
    returns:          Resized image
    """
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized_img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized_img


def save_screenshot(screenshot):
    dir_name = os.path.join(os.getcwd(),'screenshots')
    file_name = time.strftime("%Y-%m-%d-%H-%M-%S")+'.jpg'
    file_name_path = os.path.join(dir_name, file_name)
    
    if not os.path.exists(file_name_path):     #do not overwrite screenshot taken during current second

        cv2.imwrite(file_name_path, screenshot)
        



#%% SAVE SCREENSHOTS
rungame()
while True:
    screenshot = make_screenshot()
    save_screenshot(screenshot)
    
    
    
    #show window with screen contents
    cv2.imshow('window', cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB))
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break 



 
