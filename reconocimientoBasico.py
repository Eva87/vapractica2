#Ejecucion basica programa segun enunciado HOGDescriptor-LDA+Bayesiano con Gaussianas de Sklearn
from aprendizaje import *
from descriptor import *
from reconocimiento import reconocimiento, devolverResultado
from reduccionDimension import reduccionDimension

descrip = descriptorVC.creacionHOGDescriptor()
mX,mY = aprendizaje.entrenarClasificador('./train_recortadas',descrip)
ctf,xR = reduccionDimension.reducirDimensionalidadLDA(mX,mY)
listaDirectorio = os.listdir('./test_reconocimiento')
#aprendizaje.grafico(xR,mY)

for i in range(len(listaDirectorio)):
    if(listaDirectorio[i]!=".directory"):
        print(listaDirectorio[i]+": ")
        clase = reconocimiento.reconocimientoBayesianoGaussianas('./test_reconocimiento/'+listaDirectorio[i],descrip,ctf)
        devolverResultado(clase)
