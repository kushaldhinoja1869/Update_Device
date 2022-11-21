import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
#For RPi Modify this line 
import tensorflow as tf
from sys import getsizeof
import cv2
import keyboard
from time import sleep

## version 2


IMG_WIDTH = 100
IMG_HEIGHT = 100

def predict_ch(interpreter, image):
	resizedimg = cv2.resize(image, (640,480), interpolation = cv2.INTER_AREA)
	cropped_frame = image[1:150, 440:640]
	resized_cropped_frame = cv2.resize(cropped_frame, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
	resized_cropped_frame = resized_cropped_frame.astype(np.float32)
	resized_cropped_frame = resized_cropped_frame.reshape((1,100,100,3))

	interpreter.allocate_tensors()
	input_details = interpreter.get_input_details()
	output_details = interpreter.get_output_details()
	interpreter.set_tensor(input_details[0]['index'], resized_cropped_frame)
	interpreter.invoke()
	tflite_model_predictions = interpreter.get_tensor(output_details[0]['index'])
	return tflite_model_predictions
