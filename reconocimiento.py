import numpy as np
from prueba import filtradoImg

class reconocimiento():
    def reconocimientoSignal(ruta, descrip, rd):
        t = np.transpose(descrip.compute(filtradoImg(ruta)))
        print(rd.predict(t))