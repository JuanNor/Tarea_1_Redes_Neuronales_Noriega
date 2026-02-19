# %load network.py
# Los comentarios mostrados en inglés son los originales de los autores
# del código de la red neuronal, mientras que los comentarios agregados
# (español) son realizados por cuenta propia.
"""
network.py
~~~~~~~~~~
IT WORKS

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.

DESCRIPCIÓN
Este código es una entrada al funcionamiento de las redes neuronales 
por medio de feedforward. La red ajusta los pesos mediante el algoritmo
de descenso de gradiente esctocástico (SGD), calculando sus gradientes
por medio de la retropropagación (backpropagation).

Lo que se busca es minimizar la función de costo comparando la salida de
la red con la etiqueta real del dataset MNIST.
"""

#### Libraries
# Standard library
import random           #sirve para crear datos de manera "aleatoria"

# Third-party libraries
import numpy as np      #Para poder operar con vectores

class Network(object): #Es la clase que implementa a la red neuronal.
                       

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes).  #Total de capas
        self.sizes = sizes
        #self.biases sirve para inicializar los sesgos entre
        #capas ocultas y de salida
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        #Ahora se inicializan los pesos de manera aleatoria
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        #La inicialización aleatoria rompe la simetría entre neuronas,
        #lo que permite que cada una aprenda caract. dif.

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        # Para cada capa se calcula z = W*a + b (multiplicar por los pesos
        # y sumar el sesgo)
        # Posteriormente se le aplica la función sigmoide para fijar el
        # rango entre 0 y 1.
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b) 
        return a
    


    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""

        training_data = list(training_data). #Se genera una lista con
                                             #las imágenes
        n = len(training_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        for j in range(epochs):  #Se mezclan los datos para cada época.
            random.shuffle(training_data) #En específico con esta línea
            #Se evita que el modelo aprenda patrones que dependan directamente
            #del ordenamiento del dataset.
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {} : {} / {}".format(j,self.evaluate(test_data),n_test))
            else:
                print("Epoch {} complete".format(j))
            #Finalmente se manda a imprimir el avance

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        #nabla_b y nabla_w crean listas de ceros con la misma forma que
        #los pesos y sesgos para acumular gradientes.
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y) #se caculan los 
            #gradientes por medio de backprop
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        #Finalmente se actualizan los parámetros por medio del descenso 
        #de gradiente promedio. (minimizar el costo prom. del mini-batch)
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        #Se calculan los gradientes de la función de costo para cada 
        #peso y sesgo
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        # Sirve para conocer el error en la capa de salida
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            #Las siguientes líneas permiten conocer la propagación hacia
            #atrás de las capas ocultas
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        #En palabras más simples, sirve para evaluar el desempeño de 
        #la red en datos de prueba
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
        # Obtiene el índice de la neurona con mayor activación en la 
        #última capa para compararlo con la etiqueta real y, devolviendo
        #el número de aciertos

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        #La derivada se toma asumiendo la función de costo cuadrática
        # C = 1/2 |a^L - y|^2, así su derivada parcial respec. a a^L es
        #a^L - y
        return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    # sigmoid(z) = 1/(1+ e^{-z})
    return 1.0/(1.0+np.exp(-z))
    
def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))