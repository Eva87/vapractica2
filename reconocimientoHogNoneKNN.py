# Proyecto creado por Eva María Hoyo de la Cruz, TongTong Xu y Antonio Francisco Roldan Martín

# Programa segun HOGDescriptor-LDA-KNN
import guardarSalida
from aprendizaje import *
from descriptor import *
from reconocimiento import reconocimiento, devolverResultado
from reduccionDimension import reduccionDimension

class reconocimientoHOGKNN:
    def reconocimientohogknn (carpentren, carpclasif):
        descrip = descriptorVC.creacionHOGDescriptor()
        mX,mY = aprendizaje.entrenarClasificador42(carpentren,descrip)
        listaDirectorio = os.listdir(carpclasif)
        for i in range(len(listaDirectorio)):
            if(listaDirectorio[i]!=".directory"):
                print(listaDirectorio[i]+": ")
                strin=carpclasif+'/'+listaDirectorio[i]
                clase = reconocimiento.reconocimientoKNN(strin,descrip,mX,mY)
                result=devolverResultado(clase)
                guardarSalida.guardar.salidafichero(strin, result)



#estas dos lineas se pueden borrar o no segun queramos pq cuando se ejecute el main no haran nada
'''os.remove("resultado.txt")
reconocimientoHOGKNN.reconocimientohogknn('./train_recortadas', './test_reconocimiento')'''