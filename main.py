import argparse

import reconocimientoBasico
import reconocimientoHogNoneKNN
import reconocimientoHogPCAKNN
from reconocimientoHogLDAEuclideo import reconocimientoHOGLDAEuclideo
from reconocimientoHogPCAEuclideo import reconocimientoHOGPCAEuclideo

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Entrena sober train y ejecuta el clasificador sobre imgs de test')
    parser.add_argument(
        '--train_path', type=str, default="./train", help='Path al directorio de imgs de train')
    parser.add_argument(
        '--test_path', type=str, default="./test", help='Path al directorio de imgs de test')
    parser.add_argument(
        '--classifier', type=str, default="BAYES", help='String con el nombre del clasificador')

    args = parser.parse_args()

    # Cargar los datos de entrenamiento 
    # args.train_path
    strtrain = args.train_path
    strtest = args.test_path

    #Tratamiento de los datos

    # Crear el clasificador 
    if args.classifier == "BAYES":
        #detector = ...
        reconocimientoBasico.reconocimientobasico(strtrain, strtest)

    elif args.classifier == "HOGKNN":
        reconocimientoHogNoneKNN.reconocimientoHOGKNN.reconocimientohogknn(strtrain, strtest)
    elif args.classifier == "HOGPCAKNN":
        reconocimientoHogPCAKNN.reconocimientoHOGPCAKNN.reconocimientohogpcaknn(strtrain, strtest)
    elif args.classifier == "HOGPCAEuclideo":
        reconocimientoHOGPCAEuclideo.reconocimientohogpcaeuclideo(strtrain, strtest)
    elif args.classifier == "HOGLDAEuclideo":
        reconocimientoHOGLDAEuclideo.reconocimientohogdlaeuclideo(strtrain, strtest)
    else:
        raise ValueError('Tipo de clasificador incorrecto')

    # Entrenar el clasificador si es necesario ...
    # detector ...

    # Cargar y procesar imgs de test 
    # args.train_path ...

    # Guardar los resultados en ficheros de texto (en el directorio donde se 
    # ejecuta el main.py) tal y como se pide en el enunciado.





