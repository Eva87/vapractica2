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
        print("fdskjhfhkdjdshkfjfdshk")
        cadenasalidaimagen = str(datetime.datetime.now().strftime("%d%m%Y%H%M%S%f"))


        cv2.imwrite("./recortes/" + cadenasalidaimagen +".jpg", imagenCopy)
        return cadenasalidaimagen