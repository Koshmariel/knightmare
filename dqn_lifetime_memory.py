import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
#from tensorflow.keras.metrics import mean_squarred_error
import numpy as np

class ReplayBuffer():
    def __init__ (self, max_size, input_shape, n_actions):
        self.mem_size = max_size                                               #maximum size of the memory
        self.mem_cntr = 0
        self.input_shape = input_shape                                         #input shape of the evironment
        self.state_memory = np.zeros((self.mem_size, *self.input_shape), dtype=np.float32)        #stores states from the env
        self.new_state_memory = np.zeros((self.mem_size, *self.input_shape), dtype=np.float32)    #new states after taking an action
        self.action_memory = np.zeros((self.mem_size), dtype=np.uint8)               #stores taken actions
        self.reward_memory = np.zeros(self.mem_size, dtype=np.int16)                           #rewards agent receives from the env
                                                    # FLOAT ? 
        self.terminal_flags_memory = np.zeros(self.mem_size, dtype=np.uint8)      #not to keep reward memery to the next episode
        
        
    def store_transition (self, state, action, reward, state_new, done):
        index = self.mem_cntr % self.mem_size    #next available memory cluster, once memory is full start writing from the 1st cluster
        self.state_memory[index] = state
        self.new_state_memory[index] = state_new
        self.reward_memory[index] = reward
        #self.terminal_flags_memory[index] = 1 - int(done)
        self.terminal_flags_memory[index] = done
        self.action_memory[index] = action
        self.mem_cntr += 1                                        #increment memory counter
        print('mem_cntr=',self.mem_cntr)
        
    def sample_buffer(self, batch_size): # change batch_size -> number_batches
        max_mem = min(self.mem_cntr, self.mem_size) #volume of the written memory up to which batch could be taken
        
        batch = np.random.choice(max_mem, batch_size)
        
        
        states = self.state_memory[batch]
        states_new = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terminal_flags = self.terminal_flags_memory[batch]
        
        return states, actions, rewards, states_new, terminal_flags
        
        
def build_dqn (lr=0.00001, n_actions=18, input_dims=(4,320,200), fc1_dims=512, fc2_dims=512):
            
    model = Sequential()
    
    #reversing tuple because in np array shepe is (short_memory_length, gamescreen_height, gamescreen_width)
    #CNN expexts gamescreen_width, gamescreen_height, short_memory_length/channels
    model.add(Conv2D(filters=32, kernel_size=(5,5), input_shape=(input_dims[::-1]), padding ='same', activation='relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    
    model.add(Conv2D(filters=64,  kernel_size=(3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    
    model.add(Conv2D(filters=64,  kernel_size=(3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    
    model.add(Flatten())
    model.add(Dense(fc1_dims, activation = 'relu'))
    model.add(Dense(n_actions))


    model.compile(optimizer=Adam(learning_rate=lr), loss="mean_squared_error")
    





    return model


class Agent():        #lr   disount factor    random factor                epsilon decrement factor    min epsilon
    def __init__(self, lr , gamma, n_actions, batch_size, input_dims, epsilon, epsilon_dec=0.000001, epsilon_min=0.01, 
                 mem_size=1000000, replace=5, save_model_steps=10,
                 load_models=False, q_eval_fname='q_eval.h5', q_target_fname='q_target.h5'):
                #max mem size      how often          filenames for saving model
                #                copy weights 
                #             from eval to target network
                self.action_space = [i for i in range(n_actions)]              #set of available actions
#                self.n_actions = n_actions
                self.gamma = gamma
                self.epsilon = epsilon
                self.epsilon_dec = epsilon_dec
                self.epsilon_min = epsilon_min
                self.batch_size = batch_size
                self.replace = replace
                self.save_model_steps = save_model_steps
                self.q_target_model_fname = q_target_fname       #maybe not needed as instance var and should be kept as class var
                self.q_eval_model_fname = q_eval_fname           #maybe not needed as instance var and should be kept as class var
                self.learn_step_cntr = 0               # number of learning function executions
                
                self.memory = ReplayBuffer(mem_size, input_dims, n_actions)
                if load_models == False:
                    self.q_eval = build_dqn(lr, n_actions, input_dims, fc1_dims=512)
                    self.q_target = build_dqn(lr, n_actions, input_dims, fc1_dims=512)
                else:
                    self.q_eval = load_model(self.q_eval_model_fname)       #maybe not needed as instance variable and should be kept as class var
                    self.q_target = load_model(self.q_target_model_fname)   #maybe not needed as instance variable and should be kept as class var
                    print('LOADING MODELS')
                
    def replace_target_network(self): #replace weights of q_target with q_eval every REPLACE weights
        if self.replace != 0 and self.learn_step_cntr % self.replace ==0:
            self.q_target.set_weights(self.q_eval.get_weights())  
            print('replacing target network weights')
        
        
        
    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)
        
    def choose_action(self, observation):
#        state = state[np.newaxis, :]  #reshape by adding new axis
        rand_num = np.random.random()
        if rand_num < self.epsilon:
            action = np.random.choice(self.action_space)
            print('action ', action, 'random')
        else:
            state = np.array([observation], copy=False, dtype=np.float32)  #adding extra dimension
            #reshaping from (image_number,height,width) to (width,height,image_number)
            state = np.rot90(state, axes=(1,3))
            actions = self.q_eval.predict(state)
            action = np.argmax(actions)
            print('action ', action, 'predict')
            
        return action
                
                
                
                
    def learn(self):
        if self.memory.mem_cntr < self.batch_size:  # do not learn until a batch_size states received
            print('not learning, memory too small')
            return
        print('learning')
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)
        
        self.replace_target_network()
        self.save_models()
                
        
#        action_values = np.array(self.action_space, dtype=np.int8)  #[0, 1, 2, 3...]
        #action_indices = np.dot(action, action_values)  #one hot encoding -> integer encoding
        
        
        #reshaping from (image_number,height,width) to (width,height,image_number)
        state = np.rot90(state, axes=(1,3))
        new_state = np.rot90(new_state, axes=(1,3))
        q_eval = self.q_eval.predict(state)
        q_next = self.q_eval.predict(new_state)
        
        
        q_next[done] = 0.0 #if terminal state then future reward = 0

        
        #q_target = q_eval.copy()
        
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        
        q_target = q_eval[:]   #values of not taken actions
                     #copy
        
                        
        q_target[batch_index, action] = reward + self.gamma * np.max(q_next, axis=1)
        #value of taken action                    discount     max value of next state

        self.q_eval.train_on_batch(state, q_target)             #passes the sequnce of states through the eval network,
                                                                #computes the outputs and takes the dela with q_targets
                                                                #and dones the mean squared error loss
        
        self.epsilon = self.epsilon - self.epsilon_dec if self.epsilon > self.epsilon_min else self.epsilon_min
        print('epsilon=', self.epsilon)
        
        self.learn_step_cntr += 1
        print('learn_step_cntr=',self.learn_step_cntr)
        
    def save_models(self):
        if self.save_model_steps != 0 and self.learn_step_cntr % self.save_model_steps ==0:
            self.q_eval.save('q_eval_step_'+str(self.learn_step_cntr)+'.h5')
            self.q_target.save('q_target_step_'+str(self.learn_step_cntr)+'.h5')
            print('SAVING MODELS')
"""  #moved to Agent.init
    def load_models(self, q_eval_model_fname, q_target_model_fname):
        self.q_eval = load_model(self.q_eval_model_fname)
        self.q_target = load_model(self.q_target_model_fname)
        print('LOADING MODELS')
"""
        
        