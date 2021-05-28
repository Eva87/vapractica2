Salida: resultado.txt  --> Es eliminado al comienzo de la ejecucion con lo que desaparecera de una ejecucion a la siguiente, lo mismo pasa con las carpetas creadas
En el ejercicio del video a la salida de los tres se crea un video con los fotogramas modificados con las señales detectadas


ejercicio 1 
python main.py -–train_path ./train_recortadas --test_path ./test_reconocimiento -–classifier BAYES

ejercicio 2 alternativas
python main.py -–train_path ./train_recortadas --test_path ./test_reconocimiento -–classifier HOGKNN
python main.py -–train_path ./train_recortadas --test_path ./test_reconocimiento -–classifier HOGPCAKNN
python main.py -–train_path ./train_recortadas --test_path ./test_reconocimiento -–classifier HOGPCAEuclideo
python main.py -–train_path ./train_recortadas --test_path ./test_reconocimiento -–classifier HOGLDAEuclideo

ejercicio 3
de la practica 1 con mser y la alternativa ambas con el clasificador de 42 señales y de 7 señales 
python main.py -–train_path ./train_recortadas --test_path ./test_reconocimiento -–classifier 42MSER
python main.py -–train_path ./train_recortadas --test_path ./test_reconocimiento -–classifier 7MSER
python main.py -–train_path ./train_recortadas --test_path ./test_reconocimiento -–classifier 42Alternativa
python main.py -–train_path ./train_recortadas --test_path ./test_reconocimiento -–classifier 7Alternativa

ejercicio 3 video
python main.py -–train_path ./train_recortadas --test_path ./test_reconocimiento -–classifier 42altervideo
python main.py -–train_path ./train_recortadas --test_path ./test_reconocimiento -–classifier 42mservideo
python main.py -–train_path ./train_recortadas --test_path ./test_reconocimiento -–classifier 42bayesvideo
