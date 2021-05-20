from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class reduccionDimension():

    def reducirDimensionalidadLDA(X, y):
        clf = LinearDiscriminantAnalysis()
        tX = clf.fit_transform(X, y)
        return clf, tX