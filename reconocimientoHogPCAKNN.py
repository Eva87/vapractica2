# Proyecto creado por Eva María Hoyo de la Cruz, TongTong Xu y Antonio Francisco Roldan Martín

# Programa segun HOGDescriptor-PCA-KNN
import guardarSalida
from aprendizaje import *
from descriptor import *
from reconocimiento import reconocimiento, devolverResultado
from reduccionDimension import reduccionDimension

class reconocimientoHOGPCAKNN:
    def reconocimientohogpcaknn (carpentren, carpclasif):

        descrip = descriptorVC.creacionHOGDescriptor()
        mX,mY = aprendizaje.entrenarClasificador(carpentren,descrip)
        xR = reduccionDimension.reducirDimensionalidadPCA(mX,mY)
        '''aprendizaje.grafico(xR,mY)'''
        listaDirectorio = os.listdir(carpclasif)

        for i in range(len(listaDirectorio)):
            if(listaDirectorio[i]!=".directory"):
                print(listaDirectorio[i]+": ")
                strin=carpclasif+'/'+listaDirectorio[i]
                clase = reconocimiento.reconocimientoKNN(strin,descrip,xR,mY)
                result=devolverResultado(clase)
                guardarSalida.guardar.salidafichero(strin, result)


#estas dos lineas se pueden borrar o no segun queramos pq cuando se ejecute el main no haran nada
'''os.remove("resultado.txt")
reconocimientoHOGPCAKNN.reconocimientohogpcaknn('./train_recortadas', './test_reconocimiento')'''