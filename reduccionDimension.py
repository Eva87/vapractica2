from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA

class reduccionDimension():

    def reducirDimensionalidadLDA(X, y):
        clf = LinearDiscriminantAnalysis()
        tX = clf.fit_transform(X, y)
        return clf, tX

    def reducirDimensionalidadPCA(X,y):
        pca = PCA()
        tX = pca.fit_transform(X,y)
        return tX
