# Proyecto creado por Eva María Hoyo de la Cruz, TongTong Xu y Antonio Francisco Roldan Martín
import shutil

import deteccionAlternativa
import guardarSalida
from aprendizaje import *
from descriptor import *
from reconocimiento import reconocimiento, devolverResultado42
from reduccionDimension import reduccionDimension


class reconocimientovideo42Alternativa:
    def reconocimientovideo42alternativa(carpentren):

        if os.path.exists("./recortes"):
            if os.path.exists("./recortes"):
                shutil.rmtree("./recortes")
        os.mkdir("./recortes")
        if os.path.exists("./fotogramas"):
            if os.path.exists("./fotogramas"):
                shutil.rmtree("./fotogramas")
        os.mkdir("./fotogramas")

        descrip = descriptorVC.creacionHOGDescriptor()
        mX, mY = aprendizaje.entrenarClasificador42(carpentren, descrip)
        ctf, xR = reduccionDimension.reducirDimensionalidadLDA(mX, mY)

        capturadelvideo = cv2.VideoCapture('./video.mp4')
        fotogramaactual = 0

        while True:
            ret, imagen = capturadelvideo.read()
            if ret == True:
                nombre = "./fotogramas/fotograma" + str(fotogramaactual) + ".jpg"
                fotogramaactual += 1
                cv2.imwrite(nombre, imagen)
                deteccionAlternativa.alternativa.Alternativa(nombre)
                # cv2.imshow('nombre', imagen)

                listarecortes = os.listdir("./recortes")
                for i in range(len(listarecortes)):
                    if (listarecortes[i] != ".directory"):
                        print(listarecortes[i] + ": ")
                        strin = './recortes/' + listarecortes[i]
                        try:
                            clase = reconocimiento.reconocimientoBayesianoGaussianas(strin, descrip, ctf)
                            result = devolverResultado42(clase)
                            guardarSalida.guardar.salidafichero(strin, result)
                            # imagen2=imagen.clone()
                            if clase >= 0 and clase < 43:
                                impequenna = cv2.imread(carpentren + "/" + str(clase) + "/00000.ppm")
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
                        except:
                            print()
                    if os.path.exists("./recortes"):
                        if os.path.exists("./recortes"):
                            shutil.rmtree("./recortes")
                    os.mkdir("./recortes")
                cv2.waitKey(5)
            else:
                break
        capturadelvideo.release()
        cv2.destroyAllWindows()

        guardarSalida.guardar.guardarvideo("alternativa")


#reconocimientovideo42Alternativa.reconocimientovideo42alternativa('./train_recortadas')
