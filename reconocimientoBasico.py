# Proyecto creado por Eva María Hoyo de la Cruz, TongTong Xu y Antonio Francisco Roldan Martín
#Ejecucion basica programa segun enunciado HOGDescriptor-LDA+Bayesiano con Gaussianas de Sklearn
import guardarSalida
from aprendizaje import *
from descriptor import *
from reconocimiento import reconocimiento, devolverResultado42
from reduccionDimension import reduccionDimension

class reconocimientoBasico:
    def reconocimientobasico(carpentren, carpclasif):
        descrip = descriptorVC.creacionHOGDescriptor()
        mX,mY = aprendizaje.entrenarClasificador42(carpentren,descrip)
        ctf,xR = reduccionDimension.reducirDimensionalidadLDA(mX,mY)
        listaDirectorio = os.listdir(carpclasif)
        #aprendizaje.grafico(xR,mY)

        for i in range(len(listaDirectorio)):
            if(listaDirectorio[i]!=".directory"):
                print(listaDirectorio[i]+": ")
                strin=carpclasif+'/'+listaDirectorio[i]
                clase = reconocimiento.reconocimientoBayesianoGaussianas(strin,descrip,ctf)
                result=devolverResultado42(clase)
                guardarSalida.guardar.salidafichero(strin, result)

#estas dos lineas se pueden borrar o no segun queramos pq cuando se ejecute el main no haran nada
'''os.remove("resultado.txt")
reconocimientoBasico.reconocimientobasico('./train_recortadas', './test_reconocimiento')'''