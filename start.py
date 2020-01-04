import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import os
import random

import cv2
import numpy as np


def create_model():
  model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
  ])
  return model

model = create_model()

checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

latest = tf.train.latest_checkpoint(checkpoint_dir)
model.load_weights(latest)

drawing = False # true if mouse is pressed
mode = True # if True, draw rectangle. Press 'm' to toggle to curve
ix,iy = -1,-1

def validateModel(inpt):
    global model
    im2=cv2.resize(img,(28,28),interpolation = cv2.INTER_AREA)
    im2 = cv2.bitwise_not(im2)
    cv2.imshow('imageScaled',im2)
    b = np.asarray(im2)
    c = b.reshape(1,28,28)/255
    result = model.predict(c)
    for i in range(10):
        s = f"{i} : {round(result[0,i],2)}"
        print(s)
    

# mouse callback function
def draw_circle(event,x,y,flags,param):
    global ix,iy,drawing,mode,im2

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y

    elif event == cv2.EVENT_MOUSEMOVE:      
        if drawing == True:
            cv2.circle(img,(x,y),15,(0),-1)
            
    elif event == cv2.EVENT_RBUTTONDOWN:      
        img.fill((255))

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        validateModel(im2)
        print("stopped")
            
img = np.ones((512,512,1), np.uint8)*255
im2 = np.zeros((28,28,1),np.uint8)
cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_circle)

while(1):
    cv2.imshow('image',img)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('m'):
        mode = not mode
    elif k == 27:
        break

cv2.destroyAllWindows()
