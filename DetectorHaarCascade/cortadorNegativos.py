import cv2 #OpenCV
import os #Librería para acceder a funciones del sistema operativo

i = 0
tamanioX = 50
tamanioY = 50
paso = 16
ancho = 1200
alto = 720

#Se carga el fondo de escena en la variable imag
imagNombre = 'FondoDeEscena.jpg'
imag = cv2.imread(imagNombre)

#Se crea la carpeta MuestrasNegativas
os.mkdir('MuestrasNegativas')

#La imagen es recorrida a lo alto y largo 
for y in range(0, alto - tamanioY + paso, paso): 
    for x in range(0, ancho - tamanioX + paso, paso): 
        #Se recortan áreas del tamaño definido
        muestra = imag[y:y+tamanioX, x:x+tamanioX]
        #Las áreas recortadas se almacenan en la carpeta creada
        cv2.imwrite(f'MuestrasNegativas/{format(i)}.jpg', muestra)
        i += 1


