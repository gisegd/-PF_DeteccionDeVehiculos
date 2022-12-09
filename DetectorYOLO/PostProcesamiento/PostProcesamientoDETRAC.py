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
from decimal import Decimal
#Librería para conversión de formatos YOLO
import pybboxes as pbx

#Directorios del DataSet
dataSet_anotaciones='PostProcesamiento/MVI_40191.xml' 
dataSet_imagenes='img00012.jpg'
detecciones = 'PostProcesamiento/img00012.txt'

#Tamaño de la imagen (960x540 - Secuencia 40191)
alto = 540 
ancho = 960 
#Recorte de la parte inferior y superior (medido desde arriba)
recorteYsup = 75
recorteYinf = 350
#Recorte del Dataset carril izquierdo
recorteX = 525

#Se analiza como XML y se apunta al tag raíz
tree = ET.parse(dataSet_anotaciones)
root = tree.getroot()

frames = root.findall('frame')
nImagen = 12
for j in range(nImagen-1, len(frames), 1): #Recorro cada frame del DETRAC con paso 10
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

    imagen = CV.imread(dataSet_imagenes)

    #Se recorre la lista de ubicaciones de los vehículos
    for i in range(len(esquinas)):  
        #Se dibuja cada rectangulo sobre la imagen
        #Usando: cv2.rectangle(image, start_point, end_point, color, thickness)
        #Color en BGR, grosor de linea en pixeles
        CV.rectangle(imagen, (esquinas[i][0], esquinas[i][1]), (esquinas[i][2], esquinas[i][3]), (0, 255, 0), 2) #Verde
        CV.putText(imagen, 'Vehiculo ', (esquinas[i][0], esquinas[i][3]+15), CV.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    
    #Visualización de las detecciones en formato YOLO
    auto_Rects = []
    autosYolo = []
    yoloFormat = open(detecciones, "r")
    autosYolo = yoloFormat.read().splitlines()

    for auto in range (0,len(autosYolo)):
        autoCoord = autosYolo[auto].split()
        #Cada línea es un auto, se consideran las columnas 1-4
        autoYolo = (Decimal(autoCoord[1]), Decimal(autoCoord[2]), Decimal(autoCoord[3]), Decimal(autoCoord[4]))
        size = ancho, alto  # WxH of the image
        convToVOC = pbx.convert_bbox(autoYolo, from_type="yolo", to_type="voc", image_size=size)

        #convToVOC[0],convToVOC[1],convToVOC[2],convToVOC[3]
        #   xmin        ymin          xmax        ymax
        if(convToVOC[1] < recorteYsup-5 or convToVOC[3] > recorteYinf+5 or convToVOC[0] < recorteX):
            continue
        auto_Rects.append((convToVOC[0],convToVOC[1],convToVOC[2]-convToVOC[0],convToVOC[3]-convToVOC[1]))

    for (fX, fY, fW, fH) in auto_Rects:
        CV.rectangle(imagen, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 4) #Rojo
        CV.putText(imagen, 'Vehiculo ', (int(fX), int(fY)-5), CV.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

    CV.imshow('Imagen', imagen)

    break

CV.waitKey(0)
CV.destroyAllWindows()