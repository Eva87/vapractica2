# Proyecto creado por Eva María Hoyo de la Cruz, TongTong Xu y Antonio Francisco Roldan Martín
import deteccionAlternativa
import deteccionMSER
import guardarSalida
from aprendizaje import *
from descriptor import *
from reconocimiento import reconocimiento, devolverResultado7
from reduccionDimension import reduccionDimension
from clasificador import clasificador

class reconocimientop7Alternativa:
    def reconocimientop7alternativa(carpentren, carpclasif):
        descrip = descriptorVC.creacionHOGDescriptor()
        mX,mY = aprendizaje.entrenarClasificador7(carpentren,descrip)
        ctf,xR = reduccionDimension.reducirDimensionalidadLDA(mX,mY)
        listaDirectorio = os.listdir(carpclasif)
        #aprendizaje.grafico(xR,mY)

        for i in range(len(listaDirectorio)):
            if(listaDirectorio[i]!=".directory"):
                print(listaDirectorio[i]+": ")
                strin=carpclasif+'/'+listaDirectorio[i]
                deteccionAlternativa.alternativa.Alternativa(strin)

        listarecortes=os.listdir("./recortes")
        for i in range(len(listarecortes)):
            if(listarecortes[i]!=".directory"):
                print(listarecortes[i]+": ")
                strin='./recortes/'+listarecortes[i]
                clase = reconocimiento.reconocimientoBayesianoGaussianas(strin,descrip,ctf)
                result=devolverResultado7(clase)
                guardarSalida.guardar.salidafichero(strin, result)

#estas dos lineas se pueden borrar o no segun queramos pq cuando se ejecute el main no haran nada
#os.remove("resultado.txt")
#reconocimientop7Alternativa.reconocimientop7alternativa('./train_recortadas', './test_reconocimiento')