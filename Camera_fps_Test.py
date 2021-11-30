# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 10:10:08 2021

@author: ander

Camera FPS Test
Comprobar la tasa maxima de fps en la Jetson Xavier
"""

import cv2
import os
import time
import numpy as np
import matplotlib.pyplot as plt

##  Definir objeto de la camara
print("Generar Objeto de la camara")
if os.name=='nt':
    cam = cv2.VideoCapture(0)   # Windows
elif os.name =='posix':
    cam = cv2.VideoCapture(0,cv2.CAP_V4L) # Linux
else:
    print("Error al crear objeto de la camara")  
    
    # Definir la resolucion de la camara (Variable)
cam.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

## Propiedades del texto en pantalla
font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (2,475)
fontScale              =  0.4 
fontColor              = (255,255,255)
lineThickness          = 1

time_array=np.array(0)
while True:
    try:
        Camera_start_time=time.time()
        
        #Capturar imagen Desde camara 
        
        cv_flag ,image = cam.read()
        
        time.sleep(0.005)
        
        camera_end_time =time.time()-Camera_start_time-0.005;
        
        time_array=np.append(time_array,camera_end_time*1000)
        
        print('Tiempo de toma de imagen: {:.2f}'.format(camera_end_time*1000))
        
        image_text='Tasa de FPS: {:.2f}'.format(1/camera_end_time)
        
        image=cv2.putText(image,image_text,
                    bottomLeftCornerOfText, 
                    font, 
                    fontScale,
                    fontColor,
                    lineThickness)
        
        cv2.imshow('YoloV4 Output',image)
        k=cv2.waitKey(1)
        
        if k%256==32:
            print('Interrumpido por el Usuario \n\n ## Fin del programa ##')
            break

    except:
        cam.release()
        cv2.destroyAllWindows()
        print("Excepcion!!!")
        break


##  Finalizarla Ejecucion del programa
cam.release()
cv2.destroyAllWindows()

##  Plot de los tiempos de ejecucion
plt.plot(time_array[2:time_array.size])