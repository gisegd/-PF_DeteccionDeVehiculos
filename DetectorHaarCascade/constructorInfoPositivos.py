#Documentacion de Python:
#"The xml.etree.ElementTree module implements a simple and efficient API for parsing and creating XML data.""
#https://docs.python.org/3/library/xml.etree.elementtree.html
import xml.etree.ElementTree as ET
import cv2 as CV #OpenCV
#Librería para expresiones regulares
import re
#Librerías para acceder a funciones del sistema operativo y operar sobre patrones de nombres de archivos
import glob, os 

carpetaAnotaciones = 'Anotaciones_Imagenes_Entrenamiento'
carpetaImagenes = 'Imagenes_Entrenamiento'

texto = ''
''' Formato de info.dat:
img/img1.jpg  1  140 100 45 45
img/img2.jpg  2  100 200 50 50   50 30 25 25
'''

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
for j in range(0, len(xmls), 10): #Se recorre la carpeta de anotaciones
    x=x+1
    if x==301: break #Se detiene en la imagen número 300

    #Se analiza como XML y se apunta al tag raíz
    tree = ET.parse(xmls[j])
    root = tree.getroot()
    #Array de array de 4 posiciones, cada ID -> [xmin, ymin, xmax, ymax]
    esquinas = [[]] 
    #Se recorre el árbol entre sus elementos y sub-elementos buscando el tag 'bndbox' y se cargan las esquinas
    i = 0
    for car in root.iter('object'):
        #Condición para filtrar clases de vehículos
        #if(car.find('class').text == 'motorbike' or car.find('class').text == 'van'):
            #continue
        box = car.find('bndbox')
        esquinas[i].append(int(box.find('xmin').text))
        esquinas[i].append(int(box.find('ymin').text)) 
        esquinas[i].append(int(box.find('xmax').text))
        esquinas[i].append(int(box.find('ymax').text))
        i += 1
        esquinas.append([])
    esquinas.pop(i)
    print(str(j)+':'+str(esquinas))

    #Se inicializa la línea que contiene a la imagen y su número de objetos [img/img1.jpg  1] 
    texto = texto + imagenes[j] + ' ' + str(len(esquinas)) + ' '

    #Se carga la imagen
    imagen = CV.imread(imagenes[j])
    #Se recorre la lista de ubicaciones de los vehículos
    for i in range(len(esquinas)):  
        texto = texto + str(esquinas[i][0]) + ' ' + str(esquinas[i][1]) + ' ' + str(esquinas[i][2] - esquinas[i][0]) + ' ' + str(esquinas[i][3] - esquinas[i][1]) + ' '
        #                          xmin                     ymin                                  xmax - xmin                                  ymax - ymin

    texto = texto + '\n'
    
#Creación del archivo pos.txt
f = open('pos.txt', 'w')
f.write(texto)
f.close()
