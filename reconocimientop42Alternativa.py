# Proyecto creado por Eva María Hoyo de la Cruz, TongTong Xu y Antonio Francisco Roldan Martín
import deteccionAlternativa
import deteccionMSER
import guardarSalida
from aprendizaje import *
from descriptor import *
from reconocimiento import reconocimiento, devolverResultado42
from reduccionDimension import reduccionDimension
from clasificador import clasificador

class reconocimientop1Alternativa:
    def reconocimientop1alternativa(carpentren, carpclasif):
        descrip = descriptorVC.creacionHOGDescriptor()
        mX,mY = aprendizaje.entrenarClasificador42(carpentren,descrip)
        ctf,xR = reduccionDimension.reducirDimensionalidadLDA(mX,mY)
        listaDirectorio = os.listdir(carpclasif)
        #aprendizaje.grafico(xR,mY)

        for i in range(len(listaDirectorio)):
            if(listaDirectorio[i]!=".directory"):
                print(listaDirectorio[i]+": ")
                strin=carpclasif+'/'+listaDirectorio[i]
                deteccionAlternativa.alternativa.Alternativa(strin)
        try:
            os.mkdir("./recortes")
            print()
        except:
            # print("ya esta creado")
            print()
        listarecortes=os.listdir("./recortes")
        for i in range(len(listarecortes)):
            if(listarecortes[i]!=".directory"):
                print(listarecortes[i]+": ")
                strin='./recortes/'+listarecortes[i]
                clase = reconocimiento.reconocimientoBayesianoGaussianas(strin,descrip,ctf)
                result=devolverResultado42(clase)
                guardarSalida.guardar.salidafichero(strin, result)

#estas dos lineas se pueden borrar o no segun queramos pq cuando se ejecute el main no haran nada
#os.remove("resultado.txt")
reconocimientop1Alternativa.reconocimientop1alternativa('./train_recortadas', './test_reconocimiento')