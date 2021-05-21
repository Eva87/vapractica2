#Ejecucion basica programa segun enunciado HOGDescriptor-LDA+Bayesiano con Gaussianas de Sklearn
from prueba import *
from descriptor import *
from reconocimiento import reconocimiento
from reduccionDimension import reduccionDimension

descrip = descriptor.creacionHOGDescriptor()
mX,mY = aprendizaje.entrenarClasificador('./train_recortadas',descrip)
ctf,xR = reduccionDimension.reducirDimensionalidadLDA(mX,mY)
clase = reconocimiento.reconocimientoSignal('./test_reconocimiento/32-00005.ppm',descrip,ctf)
