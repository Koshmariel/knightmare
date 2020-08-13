
#Limit to CPU
#import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = '-1'


#Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback #, ModelCheckpoint

#image dimensions
image_width = 160
image_height = 100
image_channels =1



#Prepairing the data
from keras.preprocessing.image import ImageDataGenerator
"""

train_datagen = ImageDataGenerator(rotation_range=10,
                                   rescale=1./255,
                                   shear_range=0.1, #rhombus
                                   zoom_range=0.1,
                                   horizontal_flip=True,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1)
"""

train_datagen = ImageDataGenerator(rescale=1./255,
                                   width_shift_range=0.06,
                                   height_shift_range=0.06)

test_datagen = ImageDataGenerator(rescale=1./255)


train_set_path = 'G:\Python\KMold\screenshots\categorized'
test_set_path = 'G:\Python\KMold\screenshots\categorized'

#        train_datagen -> test_datagen
training_set = train_datagen.flow_from_directory(train_set_path,
                                                 color_mode="grayscale",
                                                 target_size=(image_height, image_width), #same as input
                                                 batch_size=12,
                                                 class_mode='categorical')

test_set = test_datagen.flow_from_directory(test_set_path,
                                            color_mode="grayscale",
                                            target_size=(image_height, image_width),
                                            batch_size=8,
                                            class_mode='categorical')





#Building the CNN
# (Conv+BatchNorm+MaxPool+DropOut)*3 + inrease layer size
classifier = Sequential()
classifier.add(Conv2D(64, (3, 3), padding ='same', input_shape=(image_height, image_width, image_channels), activation = 'relu'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size = (2,2)))
classifier.add(Dropout(0.25))

classifier.add(Conv2D(64, (3, 3), padding ='same', activation = 'relu'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size = (2,2)))
classifier.add(Dropout(0.25))


#removing layers had no effect on prediction accuracy or speed
classifier.add(Conv2D(128, (3, 3), padding ='same', activation = 'relu'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size = (2,2)))
classifier.add(Dropout(0.25))

classifier.add(Flatten())
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 5, activation = 'softmax'))
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


#record the computation time for each epoch
import time
class TimeHistory(Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

time_callback = TimeHistory()


#Early Stop
#to prevent over fitting stop the learning after 10 epochs if val_loss value is not decreasing
earlystop = EarlyStopping(patience=5,
                          verbose=1,
                          restore_best_weights=True)

#Learning Rate Reduction
#reduce the LR when accucarcy will not increase for 2 steps
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                                            patience=2,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)


callbacks = [earlystop, learning_rate_reduction, time_callback]


#Fitting the CNN to the images
history = classifier.fit_generator(training_set,
                                   steps_per_epoch=350,
                                   epochs=20,
                                   callbacks=callbacks,
                                   verbose=1,
                                   validation_data=test_set,
                                   validation_steps=350)
    

classifier.save('game_mode_recognizer_model_simplified')


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']



epochs = range(1, len(acc) + 1)


import matplotlib.pyplot as plt
#drawing Training and Test loss

plt.plot(epochs, loss, 'b', label='Train loss')

plt.plot(epochs, val_loss, 'b--', label='Test loss')

plt.title('Training and Test loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(fontsize='x-small')

plt.show()

plt.clf()   # clear figure


#drawing Training and Test accuracy
plt.plot(epochs, acc, 'b', label='Train acc')

plt.plot(epochs, val_acc, 'b--', label='Test acc')

plt.title('Training and Test accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
#plt.ylim(0, 80)
plt.legend(fontsize='x-small')

plt.show()

plt.clf()   # clear figure


times=time_callback.times
from statistics import mean
print('Average epoch computation time for model is: {:.2f}'.format(mean(times)))



from findFilesInFolder import findFilesInFolder
import random
import cv2
#import numpy as np
#import time

game_modes = {
            0: "0 Intro",
            1: "1 Game menu",
            2: "2 Level intro",
            3: "3 gameplay",
            4: "4 Credits"
            }    


dir_name = 'G:\Python\KMold\screenshots\categorized'
extension='.jpg'
pathList = []
pathList = findFilesInFolder(dir_name, pathList, extension, True)
sample = random.choice(pathList)
screenshot = cv2.imread(sample, cv2.IMREAD_GRAYSCALE)

if len(screenshot.shape) == 3:                                   #check if the image is color
    screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2GRAY)
screenshot = cv2.resize(screenshot, (image_width, image_height), interpolation = cv2.INTER_AREA)
screenshot = screenshot / 255.
screenshot = screenshot.reshape((1, screenshot.shape[0], screenshot.shape[1], 1)) #add dimension to transform array into a batch
start_time=time.time()
classifier_output = classifier.predict(screenshot)
end_time=time.time()
classifier_output = classifier_output[0]                    #decrease dimensions
category=max(range(len(classifier_output)), key=classifier_output.__getitem__) #much faster then argmax
plt.imshow(screenshot[0,:,:,0])
plt.show()
    
print ("Game mode is ", game_modes[category])
print (f"Predicted in {end_time - start_time} sec")