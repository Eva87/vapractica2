# Programa segun HOGDescriptor-PCA-KNN
from aprendizaje import *
from descriptor import *
from reconocimiento import reconocimiento, devolverResultado
from reduccionDimension import reduccionDimension

descrip = descriptorVC.creacionHOGDescriptor()
mX,mY= aprendizaje.entrenarClasificador('./train_recortadas',descrip)
xR = reduccionDimension.reducirDimensionalidadPCA(mX,mY)

zz = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]

for i in range (len(xR)):
    punto = [xR[i][0],xR[i][1]]
    zz[mY[i]].append(punto)

centrosX = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
centrosY = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

for j in range(len(zz)):
    aux = np.mean(zz[j],axis=0)
    print(aux)

    centrosX[j] = aux[0]
    centrosY[j] = aux[1]

print(centrosX)
print(centrosY)

listaDirectorio = os.listdir('./test_reconocimiento')
'''clase = reconocimiento.reconocimientoEuclideo('38-00004.ppm', descrip, centrosX,centrosY)
devolverResultado(clase)'''


for i in range(len(listaDirectorio)):
    if(listaDirectorio[i]!=".directory"):
        print(listaDirectorio[i]+": ")
        clase = reconocimiento.reconocimientoEuclideo('./test_reconocimiento/'+listaDirectorio[i],descrip,centrosX,centrosY)
        devolverResultado(clase)

for x, y in zip(centrosX, centrosY):
    rgb = np.random.rand(3,)
    plt.scatter(x, y, c=[rgb])

plt.show()


