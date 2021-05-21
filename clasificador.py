import cv2
from sklearn.neighbors import KNeighborsClassifier
class clasificador():

    def clasificadorKNN(X,y,t):
        neigh = KNeighborsClassifier(n_neighbors=3)
        neigh.fit(X, y)
        return neigh.predict(t)

    def clasificadorBayesianoGaussianas(rd,t):
        return rd.predict(t)


