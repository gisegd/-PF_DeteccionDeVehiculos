import os #Librería para acceder a funciones del sistema operativo

#Reemplazar raíz con el directorio donde se aloja la carpeta de proyecto  
raiz = 'C:/Users/Administrator/'
carpeta = 'DetectorHaarCascade/MuestrasNegativas/'
string = ''

#Se recorre la carpeta de muestras negativas generando un string con los nombres de los archivos
cantidadMuestras = len(os.listdir(raiz+carpeta))
for i in range(0, cantidadMuestras, 1):
    string = string + raiz + carpeta + str(i) + '.jpg\n'

#Creación del archivo pos.txt
f = open('infoNegativos.txt', 'w')   
f.write(string)
f.close()
