# Programa segun HOGDescriptor-LDA-KNN
import guardarSalida
from aprendizaje import *
from descriptor import *
from reconocimiento import reconocimiento, devolverResultado
from reduccionDimension import reduccionDimension

class reconocimientoHOGKNN:
    def reconocimientohogknn (carpentren, carpclasif):
        descrip = descriptorVC.creacionHOGDescriptor()
        mX,mY = aprendizaje.entrenarClasificador(carpentren,descrip)
        listaDirectorio = os.listdir(carpclasif)

        for i in range(len(listaDirectorio)):
            if(listaDirectorio[i]!=".directory"):
                print(listaDirectorio[i]+": ")
                strin=carpclasif+'/'+listaDirectorio[i]
                clase = reconocimiento.reconocimientoKNN(strin,descrip,mX,mY)
                result=devolverResultado(clase)
                guardarSalida.guardar.salidafichero(strin, result)



reconocimientoHOGKNN.reconocimientohogknn('./train_recortadas', './test_reconocimiento')