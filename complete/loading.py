from keras.preprocessing.image import ImageDataGenerator 
from keras.models import Sequential 
from keras.layers import Conv2D, MaxPooling2D 
from keras.layers import Activation, Dropout, Flatten, Dense 
from keras import backend as K
import numpy as np
import cv2
import warnings 
def fxn():
    warnings.warn("deprecated", DeprecationWarning)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

    
img_width, img_height = 28, 28

train_data_dir = 'a_train'
validation_data_dir = 'a_val'
nb_train_samples = 50
nb_validation_samples = 16 
epochs = 10
batch_size = 16

if K.image_data_format() == 'channels_first': 
	input_shape = (3, img_width, img_height) 
else: 
	input_shape = (img_width, img_height, 3) 


def ani():
   model = Sequential() 
   model.add(Conv2D(32, (2, 2), input_shape = input_shape)) 
   model.add(Activation('relu')) 
   model.add(MaxPooling2D(pool_size =(2, 2))) 

   model.add(Conv2D(32, (2, 2))) 
   model.add(Activation('relu')) 
   model.add(MaxPooling2D(pool_size =(2, 2))) 

   model.add(Conv2D(64, (2, 2))) 
   model.add(Activation('relu')) 
   model.add(MaxPooling2D(pool_size =(2, 2))) 

   model.add(Flatten()) 
   model.add(Dense(64)) 
   model.add(Activation('relu')) 
   model.add(Dropout(0.5)) 
   model.add(Dense(1)) 
   model.add(Activation('sigmoid')) 
   return model
model=ani()
model.load_weights('model_saved.h5')

def predict(img):
        #img=cv2.imread(img)
        x = np.expand_dims(img, axis=0)
        return model.predict(x)






