# Proyecto creado por Eva María Hoyo de la Cruz, TongTong Xu y Antonio Francisco Roldan Martín
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

def devolverResultado42(s):
    if s < 10:
        salida ="0"+str(s)
    else:
        salida = str(s)
    print(salida)
    return salida

def devolverResultado7(s):
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
    else: print("Error asignacion a clase")
    print(salida)
    return salida

