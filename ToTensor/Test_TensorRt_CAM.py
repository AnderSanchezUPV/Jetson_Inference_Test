###########################################################################################
####                                 ResNet50 en TensorRT				###
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

import matplotlib.pyplot as plt

################################# Variables GLobales ######################################
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


##	Sacar las labels
Labels = Google2Human()

##	Preparar la cámara
#vid = cv2.VideoCapture(0,cv2.CAP_V4L)
vid = cv2.VideoCapture('v4l2src device=/dev/video0 ! jpegdec ! videoconvert  ! video/x-raw, width=640, height=480 ! appsink',cv2.CAP_GSTREAMER)


## Propiedades del texto en pantalla
font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText_1 = (2,475)
bottomLeftCornerOfText_2 = (2,445)
bottomLeftCornerOfText_3 = (2,415)
fontScale              =  0.7 
fontColor              = (255,255,255)
lineThickness          = 1

# used to record the time when we processed last frame
prev_frame_time = 0
# used to record the time at which we processed current frame
new_frame_time = 0

##	Cargar Modelo en formato .plan
engine = eng.load_engine(trt_runtime, serialized_plan_fp32)
h_input, d_input, h_output, d_output, stream = inf.allocate_buffers(engine, 1, variablesTensor)

time_array=np.array(0)
##	Main Loop
current_tic=time.time()
while(True):
	try:
		# Capture the video frame
		# by frame
		previous_tic=current_tic
		current_tic=time.time()

		cv_flag, frame = vid.read()
		if not(cv_flag):  
			print('Error al capturar imagen')
			break

		# Preprocesar Imagen

		im = Cam2NetImage(frame,HEIGHT,WIDTH)

		# Inferencia
		inference_start=time.time()		
		out = inf.do_inference(engine, im, h_input, d_input, h_output, d_output, stream, 1,HEIGHT, WIDTH)
		inference_time=time.time()-inference_start

		# Post Procesar Salida
		prediction = np.argmax(out)

 		#   Definir texto en pantalla        
		ex_time=current_tic-previous_tic
		time_array=np.append(time_array,inference_time*1000)

        # image_text='FPS: {}'.format(1/ex_time)
		image_text_1='Prediction--> {}'.format(Labels[prediction])
		image_text_2='Tiempo de Ejecucion: {:.2f}'.format(ex_time*1000)    
		image_text_3='Tiempo de Inferencia: {:.4f} ms'.format(inference_time*1000)  

        #   Generar Imagen con texto en Pantalla                                           
        
		img=cv2.putText(frame,image_text_1,
                    bottomLeftCornerOfText_1, 
                    font, 
                    fontScale,
                    fontColor,
                    lineThickness)
		img=cv2.putText(img,image_text_2,
                    bottomLeftCornerOfText_2, 
                    font, 
                    fontScale,
                    fontColor,
                    lineThickness)
		img=cv2.putText(img,image_text_3,
                    bottomLeftCornerOfText_3, 
                    font, 
                    fontScale,
                    fontColor,
                    lineThickness)		
		
		# Display the resulting frame
		cv2.imshow('cam-test',img)
		#print(fps)
		
		# the 'q' button is set as the
		# quitting button you may use any
		# desired button of your choice
		if cv2.waitKey(1) & 0xFF == 32:
			break

	except: # En caso de fallo cerrar conexion a webcam y ventanas
			print('Error en loop principal')
			vid.release()
			cv2.destroyAllWindows()			
			break	

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()

# Mostrar tiempos de Ejecucion
plt.plot(time_array[2:time_array.size])
plt.show()


