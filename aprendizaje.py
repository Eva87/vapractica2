import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
class aprendizaje():

    def entrenarClasificador(ruta, descrip):
        prohibido = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 16, 17]
        peligro = [11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
        stop = [14]
        obligacion = [33,34, 35, 36, 37, 38, 39, 40]
        finRestriccion = [6,32,41,42]
        calzadaPrioridad = [12]
        ceda = [13]

        listaDirectorio = os.listdir(ruta)
        X = []
        Y = []
        for i in range(len(listaDirectorio)):
            # Almacenamos el valor de y segun tipo de señal
            if(listaDirectorio[i]!= '.DS_Store'):
                if int(listaDirectorio[i]) in prohibido:
                    z = 1
                elif int(listaDirectorio[i]) in peligro:
                    z = 2
                elif int(listaDirectorio[i]) in obligacion:
                    z = 3
                elif int(listaDirectorio[i]) == 14:
                    z = 4
                elif int(listaDirectorio[i]) == 13:
                    z = 5
                elif int(listaDirectorio[i]) == 12:
                    z = 6
                elif int(listaDirectorio[i]) in finRestriccion:
                    z = 7
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
        # Suponiendo que son 7 clases
        colors = ['navy', 'turquoise', 'darkorange', 'lime', 'magenta', 'yellow', 'red']
        for i in range(len(X)):
            aux = X[i]
            plt.plot(aux[0], aux[1], '+', color=colors[y[i] - 1])
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







