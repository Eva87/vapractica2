import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# Lectura archivo

def lecturaDatosEntrenamiento(ruta, descrip):
    prohibido = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 16, 17]
    peligro = [11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
    stop = [14]
    obligacion = [34, 35, 36, 37, 38, 39, 40]

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
            else:
                print("Ha ocurrido un error en la carga de datos")

            lCarpeta = os.listdir(ruta + '/' + listaDirectorio[i])
            for j in range(len(lCarpeta)):
                X, Y = obtenerVectorCaracteristicas(ruta + '/' + listaDirectorio[i] + '/' + lCarpeta[j], z, X, Y, descrip)

    return X,Y


def obtenerVectorCaracteristicas(ruta, valorY, vectorX, vectorY, descrip):
    img = cv2.imread(ruta)
    # Color o gris mejor ? Decidir a color por cv2.equalizeHist.compute
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # equalizado
    equ = cv2.equalizeHist(gray)
    # escalado
    dim = (30, 30)
    scl = cv2.resize(equ, dim, interpolation=cv2.INTER_AREA)
    vectorX.append(descrip.compute(scl))
    print(descrip.compute(scl))
    vectorY.append(valorY)
    return vectorX, vectorY


winSize = (20, 20)
blockSize = (10, 10)
blockStride = (5, 5)
cellSize = (10, 10)
nbins = 9
derivAperture = 1
winSigma = -1.
histogramNormType = 0
L2HysThreshold = 0.2
gammaCorrection = 1
nlevels = 64
signedGradient = True

descriptor = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                               histogramNormType, L2HysThreshold, gammaCorrection, nlevels, signedGradient)


mX,mY = lecturaDatosEntrenamiento("./train_recortadas", descriptor)
print(len(mX))
print()
print(mY)
print(len(mY))

'''
Z = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
o = np.array([1, 1, 1, 2, 2, 2])
clf = LinearDiscriminantAnalysis()
clf.fit(Z, o)
print(Z)
print(o)
print(clf.predict([[-0.8, -1]]))
'''
clf = LinearDiscriminantAnalysis()
//Error, dimensiones de X de 3
clf.fit(mX, mY)


