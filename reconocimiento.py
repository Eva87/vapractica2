import numpy as np

from aprendizaje import filtradoImg
from clasificador import clasificador


class reconocimiento():
    def reconocimientoBayesianoGaussianas(ruta, descrip, rd):
        print("Leyendo: "+ ruta)
        t = np.transpose(descrip.compute(filtradoImg(ruta)))
        resultado = clasificador.clasificadorBayesianoGaussianas(rd,t)
        iSignal = resultado[0]
        return iSignal


    def reconocimientoKNN(ruta,descrip,X,y):
        print("Leyendo: " + ruta)
        t = np.transpose(descrip.compute(filtradoImg(ruta)))
        resultado = clasificador.clasificadorKNN(X,y,t)
        iSignal = resultado[0]
        return iSignal

    def reconocimientoEuclideo(ruta,descrip,pX,pY):
        print("Leyendo: " + ruta)
        t = np.transpose(descrip.compute(filtradoImg(ruta)))
        t = t[0]
        distancia,resultado = clasificador.clasificadorEuclideo(pX,pY,t)
        return resultado

def devolverResultado(s):
    if s < 10:
        salida ="0"+str(s)
    else:
        salida = str(s)
    print(salida)
    return salida

