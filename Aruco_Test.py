import cv2
import numpy as np
from PIL import Image

#   Cargar Imagen

img=Image.open('images\Aruco\índice.jpg')
img=np.array(img)
#   Definir propiedades de los Aruco
aruco_dict=cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_1000)
arucoParameters = cv2.aruco.DetectorParameters_create()

#   Procesar Imagen
corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(img, aruco_dict, 
                                  parameters=arucoParameters)

print(ids)
#   Dibujar marcadores Aruco
frame = cv2.aruco.drawDetectedMarkers(img, corners)

cv2.imshow('Display', frame)
cv2.waitKey(0)

#   Finalizar programa
cv2.destroyAllWindows()