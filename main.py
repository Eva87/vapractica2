import argparse
import os

import reconocimientoBasico
import reconocimientoHogNoneKNN
import reconocimientoHogPCAKNN
import reconocimientop1MSER
import reconocimientoHogPCAEuclideo
import reconocimientoHogLDAEuclideo
from pathlib import Path

'''
python main.py -–train_path ./train_recortadas --test_path ./test_reconocimiento -–classifier BAYES
python main.py -–train_path ./train_recortadas --test_path ./test_reconocimiento -–classifier HOGKNN
python main.py -–train_path ./train_recortadas --test_path ./test_reconocimiento -–classifier HOGPCAKNN
python main.py -–train_path ./train_recortadas --test_path ./test_reconocimiento -–classifier HOGPCAEuclideo
python main.py -–train_path ./train_recortadas --test_path ./test_reconocimiento -–classifier HOGLDAEuclideo
python main.py -–train_path ./train_recortadas --test_path ./test_reconocimiento -–classifier Practica1MSER
python main.py -–train_path ./train_recortadas --test_path ./test_reconocimiento -–classifier Practica1Alternativa
'''
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
    if os.path.exists("resultado.txt"):
        os.remove("resultado.txt")

    # Crear el clasificador 
    if args.classifier == "BAYES":
        #detector = ...
        reconocimientoBasico.reconocimientobasico(strtrain, strtest)

    elif args.classifier == "HOGKNN":
        reconocimientoHogNoneKNN.reconocimientoHOGKNN.reconocimientohogknn(strtrain, strtest)
    elif args.classifier == "HOGPCAKNN":
        reconocimientoHogPCAKNN.reconocimientoHOGPCAKNN.reconocimientohogpcaknn(strtrain, strtest)
    elif args.classifier == "HOGPCAEuclideo":
        reconocimientoHogPCAEuclideo.reconocimientoHOGPCAEuclideo.reconocimientohogpcaeuclideo(strtrain, strtest)
    elif args.classifier == "HOGLDAEuclideo":
        reconocimientoHogLDAEuclideo.reconocimientoHOGLDAEuclideo.reconocimientohogldaeuclideo(strtrain, strtest)
    elif args.classifier == "Practica1MSER":
        reconocimientop1MSER.reconocimientop1mser(strtrain, strtest)
    else:
        raise ValueError('Tipo de clasificador incorrecto')

    # Entrenar el clasificador si es necesario ...
    # detector ...

    # Cargar y procesar imgs de test 
    # args.train_path ...

    # Guardar los resultados en ficheros de texto (en el directorio donde se 
    # ejecuta el main.py) tal y como se pide en el enunciado.





