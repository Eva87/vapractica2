# Proyecto creado por Eva María Hoyo de la Cruz, TongTong Xu y Antonio Francisco Roldan Martín
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
class aprendizaje():

    def entrenarClasificador(ruta, descrip):

        listaDirectorio = os.listdir(ruta)
        X = []
        Y = []
        lClases = []
        for i in range(len(listaDirectorio)):
            # Almacenamos el valor de y segun tipo de señal
            if(listaDirectorio[i]!= '.DS_Store'):
              z = int(listaDirectorio[i])
            else:
                print("Ha ocurrido un error en la carga de datos en carpeta: "+listaDirectorio[i])

            lCarpeta = os.listdir(ruta + '/' + listaDirectorio[i])
            for j in range(len(lCarpeta)):
                X, Y = aprendizaje.calculoxy(ruta + '/' + listaDirectorio[i] + '/' + lCarpeta[j], z, X, Y, descrip)

        return X,Y


    def calculoxy(ruta, valorY, vectorX, vectorY, descrip):
        t = np.transpose(descrip.compute(filtradoImg(ruta)))
        b = t[0]
        vectorX.append(b.tolist())
        vectorY.append(valorY)
        return vectorX, vectorY

    def grafico(X, y):
        for i in range(len(X)):
            aux = X[i]
            rgb = np.random.rand(3, )
            plt.plot(aux[0], aux[1], '+', color=rgb)
        plt.show()



#Tamaño depende se imagen
def filtradoImg(ruta):
    img = cv2.imread(ruta)
    # Color o gris mejor ? Decidir a color por cv2.equalizeHist.compute
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # equalizado
    equ = cv2.equalizeHist(gray)
    # escalado
    dim = (30, 30)
    scl = cv2.resize(equ, dim, interpolation=cv2.INTER_AREA)
    return scl







