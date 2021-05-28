# Proyecto creado por Eva María Hoyo de la Cruz, TongTong Xu y Antonio Francisco Roldan Martín

# Programa segun HOGDescriptor-LDA-KNN
import guardarSalida
from aprendizaje import *
from descriptor import *
from reconocimiento import reconocimiento, devolverResultado42
from reduccionDimension import reduccionDimension

class reconocimientoHOGKNN:
    def reconocimientohogknn (carpentren, carpclasif):
        descrip = descriptorVC.creacionHOGDescriptor()
        mX,mY = aprendizaje.entrenarClasificador42(carpentren,descrip)

        '''X_train, X_test, y_train, y_test = train_test_split(mX, mY, random_state=0)'''

        listaDirectorio = os.listdir(carpclasif)
        for i in range(len(listaDirectorio)):
            if(listaDirectorio[i]!=".directory"):
                print(listaDirectorio[i]+": ")
                strin=carpclasif+'/'+listaDirectorio[i]
                clase,classfic = reconocimiento.reconocimientoKNN(strin,descrip,mX,mY)
                result=devolverResultado42(clase)
                guardarSalida.guardar.salidafichero(strin, result)

        '''disp = plot_confusion_matrix(classfic, X_test, y_test,
                                     display_labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
                                                     19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
                                                     36, 37, 38,39],
                                     cmap=plt.cm.Blues,
                                     normalize='true')

        plt.show();'''



#estas dos lineas se pueden borrar o no segun queramos pq cuando se ejecute el main no haran nada
'''os.remove("resultado.txt")
reconocimientoHOGKNN.reconocimientohogknn('./train_recortadas', './test_reconocimiento')'''