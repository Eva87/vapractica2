#Ejecucion basica programa segun enunciado HOGDescriptor-LDA+Bayesiano con Gaussianas de Sklearn
from aprendizaje import *
from descriptor import *
from reconocimiento import reconocimiento
from reduccionDimension import reduccionDimension

descrip = descriptor.creacionHOGDescriptor()
mX,mY = aprendizaje.entrenarClasificador('./train_recortadas',descrip)
ctf,xR = reduccionDimension.reducirDimensionalidadLDA(mX,mY)
listaDirectorio = os.listdir('./test_reconocimiento')

for i in range(len(listaDirectorio)):
    if(listaDirectorio[i]!=".directory"):
        print(listaDirectorio[i]+": ")
        clase = reconocimiento.reconocimientoSignal('./test_reconocimiento/'+listaDirectorio[i],descrip,ctf)
