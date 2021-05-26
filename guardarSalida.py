# Proyecto creado por Eva María Hoyo de la Cruz, TongTong Xu y Antonio Francisco Roldan Martín

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