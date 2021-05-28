# Proyecto creado por Eva María Hoyo de la Cruz, TongTong Xu y Antonio Francisco Roldan Martín
import datetime
import os

import cv2

class guardar:

    def salidafichero(noment, nomsal):
        ficherosalida = open("resultado.txt", "a", encoding="utf-8")
        punto=0
        for i in range(len(noment)):
            if noment[i]=="/":
                punto=i
        if punto>0:
            noment=noment[punto+1:]
        ficherosalida.write(str(noment) + "; " + str(nomsal) + "\n")
        ficherosalida.close()
        return ()

    def guardarimagencarpeta(imagenCopy):
        cadenasalidaimagen = str(datetime.datetime.now().strftime("%d%m%Y%H%M%S%f"))
        cv2.imwrite("./recortes/" + cadenasalidaimagen +".jpg", imagenCopy)
        return cadenasalidaimagen

    def guardarimagencarpetaframes(imagenCopy,frame):
        cv2.imwrite("./recortes/" + frame +".jpg", imagenCopy)
        return frame

    def guardarvideo(nombre):

        listaDirec = os.listdir("./fotogramas")
        aux = cv2.imread("./fotogramas/fotograma0.jpg")
        alto, ancho, canales = aux.shape
        salidavideo = cv2.VideoWriter('salidavideosennales'+nombre+'.avi', cv2.VideoWriter_fourcc(*'DIVX'), 20.0,
                                      (ancho, alto))
        longitudfotogramas = len(listaDirec)
        i = 0
        while i < longitudfotogramas:
            strin = cv2.imread("./fotogramas/fotograma" + str(i) + ".jpg")
            salidavideo.write(strin)
            i += 1

        salidavideo.release()
