###########################################################################################
####                                 GoogleNet en TensorRT				###
###########################################################################################

# Autor: Ander Sanchez
# Fecha: 30/11/2021

######################################## Librerias ########################################
import cv2
import engine as eng
import Inference as inf


import tensorrt as trt 
import numpy as np
import time

model_dir ="./ToTensor/convert_Resnet50"
labels_dir = model_dir+"/Model_Labels.txt"
serialized_plan_fp32 = model_dir+"/resnet50.plan"
HEIGHT = 224
WIDTH = 224
variablesTensor = trt.float32
VarialbesNumpy = np.float32

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt_runtime = trt.Runtime(TRT_LOGGER)

######################################## Funciones ########################################
## Adecuar la imagen a la red
def Cam2NetImage(img,HEIGHT,WIDTH):
	#Preparar la imagen
	img = cv2.resize(img, dsize=(HEIGHT, WIDTH)) 
	img.resize()#cv2.resize(mask, size, interpolation)
	#cv2.imshow("Reajuste Foto",img)

	#Pasar la imagen a una matriz
	Data = img.astype(VarialbesNumpy)
	Data = Data.transpose()
	Data = np.expand_dims(Data, axis=0)
	return Data
## Saber que label es. Los labels están en un TXT
def Google2Human():
	f = open(labels_dir,'r')
	mensaje = f.read().splitlines() #puts the file into an array
	f.close()
	return mensaje




#Sacar las labels
Labels = Google2Human()
# Preparar la cámara
#vid = cv2.VideoCapture(0,cv2.CAP_V4L)
vid = cv2.VideoCapture('v4l2src device=/dev/video0 ! jpegdec ! videoconvert  ! video/x-raw, width=640, height=480 ! appsink',cv2.CAP_GSTREAMER)
# used to record the time when we processed last frame
prev_frame_time = 0
# used to record the time at which we processed current frame
new_frame_time = 0
print("Parada 1")
engine = eng.load_engine(trt_runtime, serialized_plan_fp32)
print("Parada 2")
h_input, d_input, h_output, d_output, stream = inf.allocate_buffers(engine, 1, variablesTensor)

while(True):

	# Capture the video frame
	# by frame
	prev_frame_time = time.time()
	ret, frame = vid.read()
	im = Cam2NetImage(frame,HEIGHT,WIDTH)
	
	out = inf.do_inference(engine, im, h_input, d_input, h_output, d_output, stream, 1,HEIGHT, WIDTH)
	salida = out[0]
	print(salida)
	#print(Labels[int(out[0])])
	# font which we will be using to display FPS
	font = cv2.FONT_HERSHEY_SIMPLEX
	# time when we finish processing for this frame
	new_frame_time = time.time()

	# Calculating the fps

	# fps will be number of frame processed in given time frame
	# since their will be most of time error of 0.001 second
	# we will be subtracting it to get more accurate result
	fps = 1/(new_frame_time-prev_frame_time)
	prev_frame_time = new_frame_time

	# converting the fps into integer
	fps = int(fps)

	# converting the fps to string so that we can display it on frame
	# by using putText function
	fps = str(fps)
	# putting the FPS count on the frame
	cv2.putText(frame, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
	
	# Display the resulting frame
	cv2.imshow('frame', frame)
	print(fps)
	
	# the 'q' button is set as the
	# quitting button you may use any
	# desired button of your choice
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()



