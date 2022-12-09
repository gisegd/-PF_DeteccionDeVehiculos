#Programa utilitario para armar una carpeta con el contenido de los datasets con un paso determinaso
#Útil para la detección con YOLO ya que se realiza desde la nube y no se puede subir el dataset completo

#Documentacion de Python:
#"The xml.etree.ElementTree module implements a simple and efficient API for parsing and creating XML data.""
#https://docs.python.org/3/library/xml.etree.elementtree.html
import xml.etree.ElementTree as ET
import cv2 
import re
import glob, os
import shutil 

numeros = re.compile(r'(\d+)')
def numericalSort(value):
    partes = numeros.split(value)
    partes[1::2] = map(int, partes[1::2])
    return partes

#Se genera la carpeta M-30-HD con paso 10 (dentro de otra con el mismo nombre para contenerla en la descompresión)
dataSet_imagenes='Imagenes_Prueba'
carpeta = 'M30HD_Paso10'
#Se crea la carpeta
os.mkdir(carpeta)
os.mkdir(carpeta+'/'+carpeta)

imagenes=sorted(glob.glob(os.path.join(dataSet_imagenes, '*.jpg')), key=numericalSort)

x=0
for j in range(0, len(imagenes), 10): 
    x=x+1
    imagen = cv2.imread(imagenes[j])
    cv2.imwrite(f'{carpeta}/{carpeta}/{imagenes[j].removeprefix(dataSet_imagenes)}', imagen)

#Se comprime la carpeta completa en formato .zip
shutil.make_archive(carpeta, 'zip', carpeta)

#Se genera la carpeta DETRAC con paso 10 (dentro de otra con el mismo nombre para contenerla en la descompresión)
dataSet_imagenes='Imagenes_Prueba_DETRAC/MVI_40191'
carpeta = 'DETRAC_Paso10'
#Se crea la carpeta
os.mkdir(carpeta)
os.mkdir(carpeta+'/'+carpeta)

imagenes=sorted(glob.glob(os.path.join(dataSet_imagenes, '*.jpg')), key=numericalSort)

x=0
for j in range(0, len(imagenes), 10): 
    x=x+1
    imagen = cv2.imread(imagenes[j])
    cv2.imwrite(f'{carpeta}/{carpeta}/{imagenes[j].removeprefix(dataSet_imagenes)}', imagen)

#Se comprime la carpeta completa en formato .zip
shutil.make_archive(carpeta, 'zip', carpeta)
