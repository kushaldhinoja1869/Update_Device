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
from datetime import datetime
import csv

########################################################################################
##          Block to Check if Updates are available on Github repo                    ##
########################################################################################

try:

	os.system("curl -LO https://raw.githubusercontent.com/kushaldhinoja1869/Update_Device/main/available_version.txt")
	print("New Version Available")
	available_version = int(open("available_version.txt", "r").read())
	current_version = int(open("current_version.txt", "r").read())

except:
	print("Error in finding a new version, going ahead with current version")
	current_version = int(open("current_version.txt", "r").read())
	available_version = int(open("current_version.txt", "r").read())
	pass

if(current_version < available_version):
	#download and replace the following files
	#delete local data.py and download new data.py
	print("Avaliable version > Current Version : Updating....")
	try:
		#os.system("del data.py")
		os.system("curl -LO https://raw.githubusercontent.com/kushaldhinoja1869/Update_Device/main/channels.py")
		print("updated channels.py")
	except:
		print("error in updating channels, passing")
		pass
	#delete local model.tflite and download model.tflite
	try:
		#os.system("del model.tflite")
		os.system("curl -LO https://github.com/kushaldhinoja1869/Update_Device/raw/main/model.tflite")
		print("updated model.tflite")
	except:
		print("error in updating model, passing")
		pass

	try:
		#os.system("del data.py")
		os.system("curl -LO https://raw.githubusercontent.com/kushaldhinoja1869/Update_Device/main/prediction_module.py")
		print("updated prediction_module.py")
	except:
		print("error in updating prediction module, passing")
		pass

	#Ater Updating
	#delete current version
	##################################IF availble_version.txt is there on local
	os.system("del current_version.txt")
	#rename available version as current version
	os.system('ren "available_version.txt" "current_version.txt"')



########################################################################################
##          Block end											                    ##
########################################################################################





import channels
import prediction_module

chlist = channels.chlist


TF_LITE_MODEL_FILE_NAME = "model.tflite"

#for RPi Modify this line
interpreter = tf.lite.Interpreter(model_path = TF_LITE_MODEL_FILE_NAME)

cap = cv2.VideoCapture(1)

i=1
while True:
		ret,frame = cap.read()
		cv2.imwrite(str(i)+ "frame.jpg",frame)
		prediction = prediction_module.predict_ch(interpreter, frame)
		predicted_class = prediction.argmax()
		#print(predicted_class)
		if(prediction[0][predicted_class] > 0.9):
			print(chlist[predicted_class])
			print(prediction[0][predicted_class])
			channel_name = chlist[predicted_class]
			channel_prob = prediction[0][predicted_class]

		else: 
			channel_name = "Unknown"
			channel_prob = prediction[0][predicted_class]
			print("Unknown")

		
		date_time = datetime.now()
		newrow = [str(date_time), str(channel_name), str(channel_prob)]
		with open("stored_data.csv","a",newline ="") as f:
			writer = csv.writer(f)
			writer.writerow(newrow)
			f.close()

		i=i+1
		sleep(1)






