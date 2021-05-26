# Proyecto creado por Eva María Hoyo de la Cruz, TongTong Xu y Antonio Francisco Roldan Martín

import deteccionMSER
import guardarSalida
from aprendizaje import *
from descriptor import *
from reconocimiento import reconocimiento, devolverResultado
from reduccionDimension import reduccionDimension
from clasificador import clasificador

class reconocimientop1MSER:
    def reconocimientop1mser(carpentren, carpclasif):
        descrip = descriptorVC.creacionHOGDescriptor()
        mX,mY = aprendizaje.entrenarClasificador7(carpentren,descrip)
        ctf,xR = reduccionDimension.reducirDimensionalidadLDA(mX,mY)
        listaDirectorio = os.listdir(carpclasif)
        #aprendizaje.grafico(xR,mY)

        for i in range(len(listaDirectorio)):
            if(listaDirectorio[i]!=".directory"):
                print(listaDirectorio[i]+": ")
                strin=carpclasif+'/'+listaDirectorio[i]
                deteccionMSER.mser.MSER(strin)
        listaDirectorio=os.listdir("./recortes/")
        for i in range(len(listaDirectorio)):
            if(listaDirectorio[i]!=".directory"):
                print(listaDirectorio[i]+": ")
                strin=carpclasif+'/'+listaDirectorio[i]
                clase = reconocimiento.reconocimientoBayesianoGaussianas(strin,descrip,ctf)
                result=devolverResultado(clase)
                guardarSalida.guardar.salidafichero(strin, result)

#estas dos lineas se pueden borrar o no segun queramos pq cuando se ejecute el main no haran nada
#os.remove("resultado.txt")
reconocimientop1MSER.reconocimientop1mser('./train_recortadas', './test_reconocimiento')