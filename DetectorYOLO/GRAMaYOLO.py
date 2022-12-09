#Documentacion de Python:
#"The xml.etree.ElementTree module implements a simple and efficient API for parsing and creating XML data.""
#https://docs.python.org/3/library/xml.etree.elementtree.html
import xml.etree.ElementTree as ET
import cv2 #OpenCV
#Librería para expresiones regulares
import re
#Librerías para acceder a funciones del sistema operativo y operar sobre patrones de nombres de archivos
import glob, os
import imutils #Funciones de OpenCV
#Librería para conversión de formatos YOLO
import pybboxes as pbx
#Librería para soporte a copia y compresión de archivos
import shutil

#Se genera la carpeta principal
os.mkdir('YOLOEntrenamiento')

#Dimensiones de la imagen
ancho = 1200
alto = 720

#Se genera el contenido de las carpetas de Entrenamiento

os.mkdir('YOLOEntrenamiento/Entrenamiento')
os.mkdir('YOLOEntrenamiento/Entrenamiento/imagenes')
os.mkdir('YOLOEntrenamiento/Entrenamiento/labels')

carpetaAnotaciones='Anotaciones_Imagenes_Entrenamiento'
carpetaImagenes='Imagenes_Entrenamiento' 
paso = 10

texto = ''

#Bloque que ordena los nombres de los archivos de imágenes y sus anotaciones de forma descendente 
#para lograr coincidencia de índices al recorrer las carpetas (de lo contrario están ordenados alfabéticamente)
#y coincidencia entre el nombre de imagen y su índice.
numeros = re.compile(r'(\d+)')
def numericalSort(value):
    partes = numeros.split(value)
    partes[1::2] = map(int, partes[1::2])
    return partes

xmls=sorted(glob.glob(os.path.join(carpetaAnotaciones, '*.xml')), key=numericalSort)
imagenes=sorted(glob.glob(os.path.join(carpetaImagenes, '*.jpg')), key=numericalSort)

x=0
for j in range(0, len(xmls), paso): #Recorro la CARPETA xml del GRAM
    x += 1
    if x==301: break
    texto = ''
    #Se analiza como XML y se apunta al tag raíz
    tree = ET.parse(xmls[j])
    root = tree.getroot()
    #Array de array de 4 posiciones, cada ID -> [xmin, ymin, xmax, ymax]
    esquinas = [[]] 
    #Se recorre el árbol entre sus elementos y sub-elementos buscando el tag 'bndbox' y se cargan las esquinas
    i = 0
    for car in root.iter('object'):
        #Se excluyen las motos del Dataset
        if(car.find('class').text == 'motorbike'): 
            continue
        box = car.find('bndbox')
        esquinas[i].append(int(box.find('xmin').text)) 
        esquinas[i].append(int(box.find('ymin').text)) 
        esquinas[i].append(int(box.find('xmax').text))
        esquinas[i].append(int(box.find('ymax').text))
        i += 1
        esquinas.append([])
    esquinas.pop(i)

    #Se lee la imagen correpondiente al XML
    imagen = cv2.imread(imagenes[j])
    #Se copia la imagen en la carpeta imagenes
    cv2.imwrite(f'YOLOEntrenamiento/Entrenamiento/imagenes/{x}.jpg', imagen)
    
    #Se recorre la lista de ubicaciones de los vehículos
    for i in range(len(esquinas)):  
        voc_bbox = (esquinas[i][0], esquinas[i][1], esquinas[i][2], esquinas[i][3])
        size = ancho, alto
        convToYOLO = pbx.convert_bbox(voc_bbox, from_type="voc", to_type="yolo", image_size=size)
        ''' Formato YOLO: 1 archivo txt por imagen
            <object-class> <x> <y> <width> <height>
        '''
        texto = texto + '0 ' + str(convToYOLO[0]) + ' ' + str(convToYOLO[1]) + ' ' + str(convToYOLO[2]) + ' ' + str(convToYOLO[3]) 
        texto = texto + '\n'
    
    f = open('YOLOEntrenamiento/Entrenamiento/labels/'+str(x)+'.txt', 'w')
    f.write(texto)
    f.close()

#Se genera el contenido de las carpetas de Prueba (de igual forma que para la de Entrenamiento)

os.mkdir('YOLOEntrenamiento/Prueba')
os.mkdir('YOLOEntrenamiento/Prueba/imagenes')
os.mkdir('YOLOEntrenamiento/Prueba/labels')

carpetaAnotaciones='Anotaciones_Imagenes_Validacion'
carpetaImagenes='Imagenes_Validacion'
paso = 1

texto = ''

numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

xmls=sorted(glob.glob(os.path.join(carpetaAnotaciones, '*.xml')), key=numericalSort)
imagenes=sorted(glob.glob(os.path.join(carpetaImagenes, '*.jpg')), key=numericalSort)

x=0
for j in range(0, len(xmls), paso): #Recorro la CARPETA xml del GRAM
    x += 1
    if x==21: break
    texto = ''
    #Se analiza como XML y se apunta al tag raíz
    tree = ET.parse(xmls[j])
    root = tree.getroot()
    #Array de array de 4 posiciones, cada ID -> [xmin, ymin, xmax, ymax]
    esquinas = [[]] 
    #Se recorre el árbol entre sus elementos y sub-elementos buscando el tag 'bndbox' y se cargan las esquinas
    i = 0
    for car in root.iter('object'):
        #Se excluyen las motos del Dataset
        if(car.find('class').text == 'motorbike'): 
            continue
        box = car.find('bndbox')
        esquinas[i].append(int(box.find('xmin').text)) 
        esquinas[i].append(int(box.find('ymin').text)) 
        esquinas[i].append(int(box.find('xmax').text))
        esquinas[i].append(int(box.find('ymax').text))
        i += 1
        esquinas.append([])
    esquinas.pop(i)

    #Se lee la imagen correpondiente al XML
    imagen = cv2.imread(imagenes[j])
    #Se copia la imagen en la carpeta imagenes
    cv2.imwrite(f'YOLOEntrenamiento/Prueba/imagenes/{x}.jpg', imagen)
    
    #Se recorre la lista de ubicaciones de los vehículos
    for i in range(len(esquinas)):  
        voc_bbox = (esquinas[i][0], esquinas[i][1], esquinas[i][2], esquinas[i][3])
        size = ancho, alto  
        convToYOLO = pbx.convert_bbox(voc_bbox, from_type="voc", to_type="yolo", image_size=size)
        ''' Formato YOLO: 1 archivo txt por imagen
            <object-class> <x> <y> <width> <height>
        '''
        texto = texto + '0 ' + str(convToYOLO[0]) + ' ' + str(convToYOLO[1]) + ' ' + str(convToYOLO[2]) + ' ' + str(convToYOLO[3]) 
        texto = texto + '\n'
    
    #Creación del archivo de anotaciones
    f = open('YOLOEntrenamiento/Prueba/labels/'+str(x)+'.txt', 'w')
    f.write(texto)
    f.close()

#Se comprime la carpeta completa en formato .zip
shutil.make_archive('YOLOEntrenamiento', 'zip', 'YOLOEntrenamiento')