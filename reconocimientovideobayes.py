import shutil

import cv2

import deteccionMSER
import guardarSalida
from aprendizaje import *
from descriptor import *
from reconocimiento import reconocimiento, devolverResultado42
from reduccionDimension import reduccionDimension


class reconocimientovideoBayes:
    def reconocimientovideobayes(carpentren):

        if os.path.exists("./recortes"):
            if os.path.exists("./recortes"):
                shutil.rmtree("./recortes")
        os.mkdir("./recortes")
        if os.path.exists("./fotogramas"):
            if os.path.exists("./fotogramas"):
                shutil.rmtree("./fotogramas")
        os.mkdir("./fotogramas")

        descrip = descriptorVC.creacionHOGDescriptor()
        mX,mY = aprendizaje.entrenarClasificador42(carpentren,descrip)
        ctf,xR = reduccionDimension.reducirDimensionalidadLDA(mX,mY)


        capturadelvideo=cv2.VideoCapture('./videos/video2.mp4')
        fotogramaactual=0

        while True:
            ret,imagen=capturadelvideo.read()
            if ret==True:
                nombre="./fotogramas/fotograma"+str(fotogramaactual)+".jpg"
                cv2.imwrite(nombre,imagen)
                try:
                    clase = reconocimiento.reconocimientoBayesianoGaussianas(nombre, descrip, ctf)
                    if clase >= 0 and clase < 43:
                        impequenna = cv2.imread("./train_recortadas/" + str(clase) + "/00000.ppm")
                        impequenna = cv2.resize(impequenna, (90, 90), interpolation=cv2.INTER_AREA)
                        rowsp, colsp, chanelsp = impequenna.shape
                        rowst, colst, chanelst = imagen.shape
                        rowstp = rowst - rowsp
                        colstp = colst - colsp
                        for i in range(rowst):
                            for j in range(colst):
                                if i >= rowstp and j >= colstp:
                                    imagen[i][j] = impequenna[i - rowstp][j - colstp]
                        cv2.imshow('nombre', imagen)
                        cv2.imwrite(nombre, imagen)
                    fotogramaactual+=1
                    cv2.waitKey(35)
                except:
                    print()
            else:
                break
        capturadelvideo.release()
        cv2.destroyAllWindows()

reconocimientovideoBayes.reconocimientovideobayes('./train_recortadas')