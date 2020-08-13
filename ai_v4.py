# -*- coding: utf-8 -*-
#%logstart -o



"""
V2 changing to lifetime memory

V3   sometimes env fails to detect agent's death
     env_v2 passes number of lives left in lives_left, if there are no lives are left it is OK to save the model & observations
V4   same as v3, made to match dqn ver 
"""
#%% INITIALIZATION

import numpy as np
from environment_v4 import Environment
from dqn_v4 import Agent
import cv2
import time



"""DEBUG
import matplotlib.pyplot as plt   #DEBUG
plt.imshow(screen)
plt.show()
    


"""


"""
game_modes = {
                1: "1 Main menu",
                2: "2 Controls",
                3: "3 Character selection",
                4: "4 Intro",
                5: "5 Level intro",
                6: "6 Gameplay",
                7: "7 Game over",
                8: "8 Continue",
                9: "9 Loading black screen"
                }

actionspace = {             #all possible key combinations
                            0: [],
                            1: [LEFT],
                            2: [UP],
                            3: [RIGHT],
                            4: [DOWN],
                            5: [SPACE],
                            6: [LEFT,SPACE],
                            7: [UP,SPACE],
                            8: [RIGHT,SPACE],
                            9: [DOWN,SPACE],
                           10: [LEFT,UP],
                           11: [LEFT,UP,SPACE],
                           12: [LEFT,DOWN],
                           13: [LEFT,DOWN,SPACE],
                           14: [RIGHT,UP],
                           15: [RIGHT,UP,SPACE],
                           16: [RIGHT,DOWN],
                           17: [RIGHT,DOWN,SPACE]
                           }
"""
short_memory_length = 4
gamescreen_width = 320          #640/2
gamescreen_height = 200         #400/2
gamescreen_height_cropped = 182 #364/2
gamescreen_channels = 1

def screen_preprocess(screen):
    if len(screen.shape) == 3:                                   #check if the image is color
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    screen = cv2.resize(screen, (gamescreen_width, gamescreen_height_cropped), interpolation = cv2.INTER_AREA)
   # screen = screen /255.
    return screen
    """
    #DEBUG look what we will get
    import matplotlib.pyplot as plt
    screen_check = screen /255.
    plt.imshow(screen)
    plt.show()
    """

#%% MAIN
env = Environment()

n_games = 3000
agent = Agent(lr=0.0001, gamma=0.7, n_actions=18, batch_size=64, input_dims=(short_memory_length, gamescreen_height_cropped, gamescreen_width),
                epsilon=.5, epsilon_dec=0.001, epsilon_min=0.1, mem_size=2000, replace = 5, save_model_steps=50,
                load_models=True, q_eval_fname='q_eval_step_1304.h5', q_target_fname='q_target_step_740.h5',
                short_memory_length = 4, load_lives_memory_option = True)

screen = env.reset(restart_game=False)
total_rewards = []
step_num = 0
for i in range(1,n_games+1):
    #print('\nGame ',i)
    screens = []
    actions = []
    rewards = []
    dones = []
    screen = env.reset()
    #screen = env.reset(restart_game=True)
    screen= screen_preprocess(screen)
    screens.append(screen)          #storing the screnshot before the performed action
    short_memory = [screen, screen ,screen, screen]  #DEBUG, change to various length
    #short_memory_new = short_memory.copy()
    done = False
    while not done:
        step_num += 1
        print('\nGame ',i, 'Step ',step_num)
        step_start_time = time.time()
        action = agent.choose_action(short_memory)
        #screen_new, reward, done, info = env.step(action)
        screen_new, reward, done, lives_left = env.step(action)  #screen_new can be replaced with screen, named this way to emphasize that
                                                           #we store the screenshot for the nex step
        screen_new = screen_preprocess(screen_new)
        screens.append(screen_new) #storing the screenshot which is the the result of the performed action.
        actions.append(action)     #storing the performed action 
        rewards.append(reward)     #storing the reward for the performed action
        dones.append(done)         #storing the flag if the step had finished the game, almost always it will be agent's death
        print(f'Reward= {reward:.5f}')
        print('Done=',done)
        screen_new= screen_preprocess(screen_new)
        #short_memory_new.pop(0)
        #short_memory_new.append(screen_new)
        short_memory.pop(0)             # pop and append directly to the short memory
        short_memory.append(screen_new) # no need to store short_memory
        #agent.remember(screen, action, reward, short_memory_new, done)
        #short_memory = short_memory_new.copy()
        #agent.learn()  #learn between games 
        step_end_time = time.time()
        step_time = step_end_time - step_start_time
        print(f'Step time {step_time:.4f}\n')
    screens.pop() # the screenshot of the finished environment is not needed and is excessive
    agent.remember(screens, actions, rewards, dones)  #maybe dones is redundant

    learn_start_time = time.time()
    agent.learn(lives_left)  #learn between games, maybe should be moved to the end of final game
    learn_time = step_end_time - step_start_time
    print(f'Learn time {step_time:.4f}')
    total_reward = sum(rewards)
    print('Total game reward = ', total_reward)
    total_rewards.append(total_reward)
    print('Average game reward = ', sum(total_rewards)/len(total_rewards))    
        
        
        
        
        