import numpy as np
import switch as switch

from aprendizaje import filtradoImg

class reconocimiento():
    def reconocimientoSignal(ruta, descrip, rd):
        print("Leyendo: "+ ruta)
        t = np.transpose(descrip.compute(filtradoImg(ruta)))
        resultado = rd.predict(t)
        iSignal = resultado[0]

        '''with switch(iSignal) as s:
            if s.case(1,True):
                salida="Prohibido"
            if s.case(2,True):
                salida="Peligro"
            if s.case(3,True):
                salida="Obligacion"
            if s.case(4,True):
                salida="Stop"
            if s.case(5,True):
                salida="Ceda"
            if s.case(6,True):
                salida="Calzada con prioridad"
            if s.case(7,True):
                salida="Fin de restrincion"
        '''
        print(iSignal)
        return iSignal