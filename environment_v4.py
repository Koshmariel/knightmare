"""
v2   sometimes env fails to detect agent's death
     passes number of lives left in info, if there are no lives are left it is OK to save the model & observations
V3   same as v2, made to match dqn ver
V4   same as v3, made to match dqn ver
"""
# -*- coding: utf-8 -*-
#%% INITIALIZATION: imports
"""
Class simulates openai gym ebvironments.
Input:      action
Output:     observations        CNN from screenshot?
            reward              delta score, penalty if died
            done                id died or game completion
            info                ?
"""
import numpy as np
#from PIL import ImageGrab
import cv2
import time
from directkeys import PressKey, ReleaseKey, UP, LEFT, RIGHT, DOWN, CTRL, DIK_3
#import os
from game_mode_recognizer import GameModeRecognizer
from score_lives_recognizer import ScoreLivesRecognizer
import matplotlib.pyplot as plt   #DEBUG
import subprocess
import win32gui
import pyautogui




#%% CLASS Environment

class Environment():
    
    def __init__(self, autoreset = True):
        """
        Parameters
        ----------
        autoreset : bool, optional
            Autoreser the evironment if done. The default is True.

        Returns
        -------
        None.

        """
        self.actionspace = {             #all possible key combinations
                            0: [],
                            1: [LEFT],
                            2: [UP],
                            3: [RIGHT],
                            4: [DOWN],
                            5: [CTRL],
                            6: [LEFT,CTRL],
                            7: [UP,CTRL],
                            8: [RIGHT,CTRL],
                            9: [DOWN,CTRL],
                           10: [LEFT,UP],
                           11: [LEFT,UP,CTRL],
                           12: [LEFT,DOWN],
                           13: [LEFT,DOWN,CTRL],
                           14: [RIGHT,UP],
                           15: [RIGHT,UP,CTRL],
                           16: [RIGHT,DOWN],
                           17: [RIGHT,DOWN,CTRL]
                           }
        self.score_lives_recognizer = ScoreLivesRecognizer()
        
        self.game_is_running = False
        
        self.game_mode_recognizer = GameModeRecognizer()
        
        
        
    def rungame(self):
        
        
        """
        #Usefull to get a list of window names
        
        import win32gui
        
        def winEnumHandler( hwnd, ctx ):
            if win32gui.IsWindowVisible( hwnd ):
                print (hex(hwnd), win32gui.GetWindowText( hwnd ))

        win32gui.EnumWindows( winEnumHandler, None )
        """
        
        
        #import os
        #os.system(r'G:\KmareShortcut.lnk')      #not recommended

        #set a working directory to the game folder and then restore it
        #work_dir = os.getcwd()
        #km_dir=r'G:\Python\knight-mare\kmare_gold_win32'
        #os.chdir(km_dir)
        #os.startfile(r'G:\Python\knight-mare\kmare_gold_win32\Kmare.exe')
        #os.chdir(work_dir)
        
        
        #subprocess is modern way of running processes
        #subprocess.run waits for process to complete
        self.game_process = subprocess.Popen(r'G:\Python\KMold\Game\DOSBox-0.74-2\DOSBox.exe', cwd=r'G:\Python\KMold\Game\DOSBox-0.74-2')
        
        
        
        import time
        time.sleep(5)  #wait until a game will start
        
        #import win32gui
        self.hwnd = win32gui.FindWindow(None, "DOSBox 0.74-2, Cpu speed:     3000 cycles, Frameskip  0, Program:     MYTH")
        print('Launching game')
        rect = win32gui.GetWindowRect(self.hwnd) #get window size
        width = rect[2] - rect[0]
        height = rect[3] - rect[1]
        #                   Win has an invisible border of 7 pixels, but it looks like KM windows actual size is a bit larger
        win32gui.MoveWindow(self.hwnd, -4, 0, width, height, True) #move the window to the top left corner
        print('Moving game window')
        print('Waiting for the game to load')
        time.sleep(20)  #wait until a game loads to the main menu
        print('Now the game should had been loaded to the 1st menu')
        self.game_is_running = True
        #lives are checked when environment restarts to close and reopen the game if there are no lives left
        #it is faster to restart the game then to wait for credits to finish
        self.lives = 666 
        
    def make_screenshot(self):
        
        if self.hwnd:
            win32gui.SetForegroundWindow(self.hwnd)
            x, y, x1, y1 = win32gui.GetClientRect(self.hwnd)
            x, y = win32gui.ClientToScreen(self.hwnd, (x, y))
            x1, y1 = win32gui.ClientToScreen(self.hwnd, (x1 - x, y1 - y))
            screen = pyautogui.screenshot(region=(x, y, x1, y1))
            screen = np.float32(screen)
            screen = screen /255.
            return screen
        else:
            print('Window not found!')

        
        
    def reset(self, restart_game=False):
        if restart_game:
            self.game_is_running = False
        if self.game_is_running == False:
            self.rungame()
        
        if self.lives == 0: #it is faster to restart the game then to wait for credits to finish
            print('Terminating the game')
            self.game_process.kill()
            self.rungame()
        #else:
            
        self.gameplay_started = False
        
        while (not self.gameplay_started):
            screen = self.make_screenshot()
            
            game_mode = self.game_mode_recognizer.recognize(screen)
            #print('GM ', game_mode)  #DEBUG
            
            if game_mode == 1:  # 1 Intro -> 2 Game menu
                
                # 1 Intro -> 2 Game menu
                
                print('1 Intro -> 2 Game menu')
                time.sleep(1)
                PressKey(CTRL)
                print('press CTRL')
                time.sleep(0.3)
                ReleaseKey(CTRL)
                print('release CTRL')
                time.sleep(2)
                
            if game_mode == 2:   # 2 Game menu -> 3 Level intro
            
                # 2 Game menu -> 3 Level intro
                
                print('2 Game menu -> 3 Level intro')
                PressKey(DIK_3)
                print('press DIK_3')
                time.sleep(0.3)
                ReleaseKey(DIK_3)
                print('release DIK_3')
                time.sleep(2)
                
                
            if game_mode == 3:   #3 Level intro -> 4 Gameplay
            
                #3 Level intro -> 4 Gameplay
                
                print('3 Level intro -> 4 Gameplay')
                PressKey(CTRL)
                print('press CTRL')
                time.sleep(0.3)
                ReleaseKey(CTRL)
                print('release CTRL')
                time.sleep(2)
                

            # 4 Gameplay

            if game_mode == 4:
                self.gameplay_started = True
                print('Gameplay started')  #DEBUG

            # 5 Credits

            if game_mode == 5:  # Just wait to 7 Game over -> 8 Continue
                print('5 Credits. We should not be here.')
                print('Terminating the game')
                #raise Exception("Credits. We should not be here.")
                self.game_process.kill()
                self.rungame()
                self.gameplay_started = False
            """
            if game_mode == 8:  # 8 Continue -> 6 Gameplay
                print('8 Continue -> 6 Gameplay')
                PressKey(CTRL)
                print('press CTRL')
                time.sleep(0.1)
                ReleaseKey(CTRL)
            """
        self.score, self.lives = self.score_lives_recognizer.recognize(screen)
        screen=screen[0:364,:, :]  #cut the lower part with score, lives etc.
        
        return screen


   
    def step(self, action):
        
        """
        Parameters
        ----------
        action : int
            Number of action from actionspace

        Returns
        -------
        screen : image
            part of screenshot which is importand for gameplay
        delta_score : int
            reward in openai.gym. = score_next - score 
            ???  + some bonus for surviving the step
            ???  - some penalty for dying
        death_flag : bool
            done in openai.gym. True if agent dies (or completes the game in the future)
        empty_var : dict
            info in openai.gym. Empty dict. In openai.gym it contains some debuging information, but not allowed to use for learning.

        """
        
        
        
        
        for key in self.actionspace[action]:
            PressKey(key)
        time.sleep(0.25)
        
        screen = self.make_screenshot()
        screen = np.float32(screen)
        
                
        self.old_score = self.score
        self.old_lives = self.lives
        self.score, self.lives = self.score_lives_recognizer.recognize(screen)
        reward = self.score - self.old_score + 10     #for staying alive for 1 step
        
        #check if agent dies
        d_lives = self.lives - self.old_lives
        
        death_flag = 0
        if d_lives == -1:
            death_flag = 1
            reward = reward - 1000           #penalty for dying
            print('DIED, lives ',self.old_lives,' -> ',self.lives)
            self.gameplay_started = False
            #if self.lives != 0:
            #    time.sleep(1)
            
            
        empty_var = {} 
        
        for key in self.actionspace[action]:
            ReleaseKey(key)
            
        screen=screen[0:364,:, :]  #cut the lower part with score, lives etc.
        return screen, reward, death_flag, self.lives
    

    

#%% MAIN

if __name__ == '__main__':
    
    #prepare to recognize game mode
    
    game_modes = {
                    1: "0 Intro",
                    2: "1 Game menu",
                    3: "2 Controls",
                    4: "2 Level intro",
                    5: "3 gameplay",
                    6: "4 Credits",
                    }
    
    actionspace = {             #all possible key combinations
                                0: '[DO NOTHING]',
                                1: '[LEFT]',
                                2: '[UP]',
                                3: '[RIGHT]',
                                4: '[DOWN]',
                                5: '[SPACE]',
                                6: '[LEFT,SPACE]',
                                7: '[UP,SPACE]',
                                8: '[RIGHT,SPACE]',
                                9: '[DOWN,SPACE]',
                               10: '[LEFT,UP]',
                               11: '[LEFT,UP,SPACE]',
                               12: '[LEFT,DOWN]',
                               13: '[LEFT,DOWN,SPACE]',
                               14: '[RIGHT,UP]',
                               15: '[RIGHT,UP,SPACE]',
                               16: '[RIGHT,DOWN]',
                               17: '[RIGHT,DOWN,SPACE]'
                               }
    
    env = Environment()
    screen = env.reset()
    
    #time.sleep(1)
    
    screens = []
    rewards = []
    dones = []
    

    
    
    import random
    while True:
         
        action = random.randint(0,17)  #upper bound included
        #action = 0
        print('Performing action', actionspace[action])
        #action = 0
        screen_new, reward, done, info = env.step(action)
        screens.append(screen_new)
        rewards.append(reward)
        dones.append(done)
        if done:
            print('DIED, waiting')
            time.sleep(4)
            print('DIED, resuming')
            screen = env.reset()
        print('dones: ', dones[-10:])
        print('rewards: ', rewards[-10:],'\n')
    
    
    
    screen_new = screen_new /255
    plt.imshow(screen_new)
    plt.show()
    
    
    
    
