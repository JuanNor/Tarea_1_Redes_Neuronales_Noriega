"""
Este archivo sirve para entrenar la red creada en network.py,
cargando los datos de MNIST
"""
#Primero se importan los datos de mnist
import mnist_loader

#Después se separan las imágenes en: entrenamiento, validación y prueba
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
#Se genera una lista de los datos de entrenamiento
training_data = list(training_data)


import network
"""Se importa la red neuronal creada en network.py"""
#Se configura la red para tener 784 neuronas de entrada (lo cual coincide)
#con el tamaño de cada imagen (28x28 píxeles), 30 ocultas
#y 10 de salida (las de salida representan las probabilidades asociadas
#a cada dígito del 0 al 9)
net = network.Network([784, 30, 10])

net.SGD(training_data, 30, 10, 3.0, test_data=test_data) #Entrenamiento de la red
#Se utilizan 30 épocas con mini.batches de tamaño 10 y una tasa de aprendizaje
#de 3.0   