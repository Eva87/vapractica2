# Proyecto creado por Eva María Hoyo de la Cruz, TongTong Xu y Antonio Francisco Roldan Martín

import deteccionMSER
import guardarSalida
from aprendizaje import *
from descriptor import *
from reconocimiento import reconocimiento, devolverResultado42
from reduccionDimension import reduccionDimension
from clasificador import clasificador

class reconocimientop42MSER:
    def reconocimientop42mser(carpentren, carpclasif):
        descrip = descriptorVC.creacionHOGDescriptor()
        mX,mY = aprendizaje.entrenarClasificador42(carpentren,descrip)
        ctf,xR = reduccionDimension.reducirDimensionalidadLDA(mX,mY)
        listaDirectorio = os.listdir(carpclasif)
        #aprendizaje.grafico(xR,mY)

        for i in range(len(listaDirectorio)):
            if(listaDirectorio[i]!=".directory"):
                print(listaDirectorio[i]+": ")
                strin=carpclasif+'/'+listaDirectorio[i]
                deteccionMSER.mser.MSER(strin)

        listarecortes=os.listdir("./recortes")
        for i in range(len(listarecortes)):
            if(listarecortes[i]!=".directory"):
                print(listarecortes[i]+": ")
                strin='./recortes/'+listarecortes[i]
                clase = reconocimiento.reconocimientoBayesianoGaussianas(strin,descrip,ctf)
                result=devolverResultado42(clase)
                guardarSalida.guardar.salidafichero(strin, result)

        return
#estas dos lineas se pueden borrar o no segun queramos pq cuando se ejecute el main no haran nada
#os.remove("resultado.txt")
#reconocimientop42MSER.reconocimientop42mser('./train_recortadas', './test_reconocimiento')