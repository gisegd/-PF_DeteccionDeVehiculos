#Documentacion de Python:
#"The xml.etree.ElementTree module implements a simple and efficient API for parsing and creating XML data.""
#https://docs.python.org/3/library/xml.etree.elementtree.html
import xml.etree.ElementTree as ET
import cv2 as CV #OpenCV
#Librería para expresiones regulares
import re
#Librerías para acceder a funciones del sistema operativo y operar sobre patrones de nombres de archivos
import glob, os
#Librería para cálculo numérico avanzado
import numpy
#Librería para acceder a funciones del sistema
import sys

#Instanciación del clasificador entrenado
detectores = {}
detectores["vehiculos"] = CV.CascadeClassifier('Cascade6/cascade.xml')

#Directorios del DataSet
carpetaAnotaciones='Anotaciones_Imagenes_Prueba_DETRAC/MVI_40191.xml' 
carpetaImagenes='Imagenes_Prueba_DETRAC/MVI_40191'
#Sizes para el detectMultiScale
minSize = 20
maxSize = 70 
#Tamaño de la imagen (960x540 - Secuencia 40191)
alto = 540 
ancho = 960 
#Recorte de la parte inferior y superior (medido desde arriba)
recorteYsup = 80
recorteYinf = 350
#Filtro del Dataset carril izquierdo
recorteX = 525

#Bloque que ordena los nombres de los archivos de imágenes y sus anotaciones de forma descendente 
#para lograr coincidencia de índices al recorrer las carpetas (de lo contrario están ordenados alfabéticamente)
#y coincidencia entre el nombre de imagen y su índice.
numeros = re.compile(r'(\d+)')
def numericalSort(value):
    partes = numeros.split(value)
    partes[1::2] = map(int, partes[1::2])
    return partes

xmls=''
imagenes=sorted(glob.glob(os.path.join(carpetaImagenes, '*.jpg')), key=numericalSort)

#Se analiza como XML y se apunta al tag raíz
tree = ET.parse(carpetaAnotaciones)
root = tree.getroot()
frames = root.findall('frame')

nImagen= 12
#Captura del argumento de entrada - Imagen a evaluar
if (len(sys.argv) == 2):
    if (int(sys.argv[1]) > len(frames)):
        print('El DataSet sólo tiene ' +str(len(frames))+' imágenes')
        sys.exit()
    else:
        nImagen = int(sys.argv[1])
print('Evaluando la imagen '+str(nImagen))

for j in range(nImagen-1, len(frames), 10): #Se recorre el archivo de anotaciones
    #Array de array de 4 posiciones, cada ID -> [xmin, ymin, xmax, ymax]
    esquinas = [[]] 
    #Se recorre el árbol entre sus elementos y sub-elementos buscando el tag 'bndbox' y se cargan las esquinas
    i = 0
    for target in frames[j].iter('target'):
        autoid = target.get('id')
        for box in target.findall('box'):
            xmin = round(float(box.get('left')))
            ymin = round(float(box.get('top')))
            xmax = round(float(box.get('left'))+float(box.get('width')))
            ymax = round(float(box.get('top'))+float(box.get('height')))
            oclusion = False
            #if len(target.findall('occlusion')) > 0: oclusion = True #-->Sin oclusiones
            if(ymin < recorteYsup or ymax > recorteYinf or xmin < recorteX or oclusion):
                continue
            esquinas[i].append(xmin)
            esquinas[i].append(ymin)
            esquinas[i].append(xmax)
            esquinas[i].append(ymax)
            esquinas[i].append(autoid)
            i += 1
            esquinas.append([])
    esquinas.pop(i)

    imagen = CV.imread(imagenes[j])
    imagen_cascade = CV.imread(imagenes[j])
    mascaraCarriles = CV.imread('MascaraCarrilesDETRAC.png')
    threshold = CV.threshold(mascaraCarriles, 0, 1, CV.THRESH_BINARY)[1]
    #Extracción Carril izquierdo y derecho
    imagen_cascade = imagen_cascade*threshold
    #Recorte de la parte inferior y superior
    imagen_cascade = imagen_cascade[recorteYsup:recorteYinf, 0:ancho]
    
    #Se recorre la lista de ubicaciones de los vehículos
    for i in range(len(esquinas)):  
        #Se dibuja cada rectangulo sobre la imagen
        #Usando: cv2.rectangle(image, start_point, end_point, color, thickness)
        #Color en BGR, grosor de linea en pixeles
        CV.rectangle(imagen, (esquinas[i][0], esquinas[i][1]), (esquinas[i][2], esquinas[i][3]), (0, 255, 0), 2) #Verde
        CV.putText(imagen, 'Vehiculo ', (esquinas[i][0], esquinas[i][3]+15), CV.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    
    #Cascade: 
    auto_Rects = detectores["vehiculos"].detectMultiScale(
        imagen_cascade, scaleFactor=1.1, minNeighbors=30, minSize=(minSize,minSize), maxSize=(maxSize, maxSize),
        flags=CV.CASCADE_SCALE_IMAGE)

    f = 0
    for (fX, fY, fW, fH) in auto_Rects:
        CV.rectangle(imagen, (fX, fY+recorteYsup), (fX + fW, fY + recorteYsup + fH), (0, 0, 255), 4) #Rojo
        CV.putText(imagen, 'Vehiculo ', (int(fX), int(fY+recorteYsup)-5), CV.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
        f += 1

    CV.imshow('Imagen', imagen)
    break

CV.waitKey(0)
CV.destroyAllWindows()