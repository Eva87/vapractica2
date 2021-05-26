# Proyecto creado por Eva María Hoyo de la Cruz, TongTong Xu y Antonio Francisco Roldan Martín
import datetime
import cv as cv2

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

    def guardarimagencarpeta(imagensalida, variablesenl, orig, nomimageent, imagenCopy):
        cadenasalidaimagen = str(datetime.datetime.now().strftime("%d%m%Y%H%M%S%f"))
        # nomsal = nomimageent[7:12]
        nomsal = nomimageent.replace("/", "")
        nomsal = nomsal.replace("\\", "")
        nomsal = nomsal.replace(".", "")
        #cadenasalidaimagen = "./recortes" + "/" + nomimageent + cadenasalidaimagen + ".jpg"


        cv2.imwrite("./recortes/" + cadenasalidaimagen + ".jpg", imagenCopy)
        return