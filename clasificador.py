import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
class clasificador():

    def clasificadorKNN(X,y,t):
        neigh = KNeighborsClassifier(n_neighbors=3)
        neigh.fit(X, y)
        return neigh.predict(t)

    def clasificadorBayesianoGaussianas(rd,t):
        return rd.predict(t)

    def clasificadorEuclideo(xC,yC,t):
        for i in range (len(xC)):
            d = np.sqrt(((xC[i]-t[0]) ** 2)+(((yC[i]-t[1]) ** 2)))
            #Caso base
            if i==0:
                min = d
                clase = 0
            else:
                if d<=min:
                    min = d
                    clase = i
        return min,clase



