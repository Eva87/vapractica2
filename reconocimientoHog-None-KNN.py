# Programa segun HOGDescriptor-LDA-KNN
from aprendizaje import *
from descriptor import *
from reconocimiento import reconocimiento, devolverResultado
from reduccionDimension import reduccionDimension

descrip = descriptorVC.creacionHOGDescriptor()
mX,mY = aprendizaje.entrenarClasificador('./train_recortadas',descrip)
listaDirectorio = os.listdir('./test_reconocimiento')

for i in range(len(listaDirectorio)):
    if(listaDirectorio[i]!=".directory"):
        print(listaDirectorio[i]+": ")
        clase = reconocimiento.reconocimientoKNN('./test_reconocimiento/'+listaDirectorio[i],descrip,mX,mY)
        devolverResultado(clase)