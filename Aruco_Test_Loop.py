import cv2
import numpy as np



#   Definir objeto de la camara
cam = cv2.VideoCapture('v4l2src device=/dev/video2 ! jpegdec ! videoconvert  ! video/x-raw, width=1920, height=1080 ! appsink drop=true sync=false',cv2.CAP_GSTREAMER)# Ubuntu
#cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)



#   Definir propiedades de los Aruco
aruco_dict=cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_1000)
arucoParameters = cv2.aruco.DetectorParameters_create()

#   Loop Captura de Imagen via webcam
while True:
    try:
        #   Tomar Imagen
        cv_flag ,img = cam.read() 
        if not(cv_flag):  
            print('Error al capturar imagen')
            break
        #   Procesar Imagen
        #img=np.mean(img,-1)
        img=cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(img, aruco_dict, 
                                        parameters=arucoParameters)
        

        #   Dibujar marcadores Aruco
        frame = cv2.aruco.drawDetectedMarkers(img, corners)

        cv2.imshow('Display', frame)
        if cv2.waitKey(1) & 0xFF == 32:
            cam.release()
            cv2.destroyAllWindows()
            break
    except:
        print('Error en loop principal')
        cam.release()
        cv2.destroyAllWindows()        
        break

#   Finalizar programa
cam.release()
cv2.destroyAllWindows()