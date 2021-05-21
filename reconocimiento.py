import numpy as np
import switch as switch

from aprendizaje import filtradoImg
from clasificador import clasificador


class reconocimiento():
    def reconocimientoSignalBayesianoGaussianas(ruta, descrip, rd):
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

def devolverResultado(s):
    if s == 1:
        salida = "Prohibido"
    elif s == 2:
        salida = "Peligro"
    elif s == 3:
        salida = "Obligacion"
    elif s == 4:
        salida = "Stop"
    elif s == 5:
        salida = "Ceda"
    elif s == 6:
        salida = "Calzada con prioridad"
    elif s == 7:
        salida = "Fin de restrincion"
    else:
        print("Error asignaciona clase")
    print(salida)

