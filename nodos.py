import numpy as np

class Node:
    '''
    '''
    def __call__(self, x):
        return self.forward(x)
    
    def __str__(self):
        return str(self.out)
    
#---------------------------------------------------------------------------------------------------
# Empezamos a crear nuestras sub clases que tambien seran nodos
# Primero definimos el que sera nuestro nodo principal para tres de nuestros modelo
# el cual es el nodo de pre-activacion
class PreActivation(Node):
    '''
    Funcion de pre-activacion
    '''
    def __init__(self, input_size, output_size):
        # fijamos una semilla para replicar datos
        np.random.seed(42)
        self.weight = np.random.randn(input_size, output_size) # el peso o los pesos son un tensor
        self.bias = np.random.randn(output_size) # el bias es un tensor
        self.inputs = None
        
    def forward(self, x):
        self.inputs = x
        self.out = np.dot(x, self.weight) + self.bias
        return self.out
    # aqui el grad_out es el gradiente calculado en el nodo sigmoide (o en el nodo de activacion en general)
    def backward(self, grad_out):
        self.grad_weight = np.dot(self.inputs.T, grad_out) / self.inputs.shape[0]
        #self.grad_weight = grad_out * self.inputs
        self.grad_bias = np.mean(grad_out, axis=0)
        #self.grad_bias = grad_out
        return self.grad_weight, self.grad_bias
    
#---------------------------------------------------------------------------------------------------
# Definimos un nodo para la funcion de activacion Sigmoide
class Sigmoide(Node):
    '''
    Funcion de activacion sigmoide
    '''
    def forward(self, z):
        self.out = 1 / (1 + np.exp(-z))
        return self.out
    
    # aqui el grad_out es el gradiente calculado en el nodo CrossEntropy
    def backward(self, grad_out):
        grad_in = self.out * (1 - self.out)
        return grad_in * grad_out
    
    def predict(self, y_pred):
        self.pred = np.where(y_pred > 0.5, 1, 0)
        return self.pred
    
#---------------------------------------------------------------------------------------------------
# Definimos un nodo para la funcion de entropia cruzada
# La cual es necesaria para hacer el calculo del gradiente
class CrossEntropy(Node):
    '''
    Funcion objetivo
    '''
    def forward(self, y_pred, y_true):
        epsilon = 1e-15 # para evitar caer en un log(0)
        y_pred = np.clip(y_pred, epsilon, 1-epsilon)
        y_true = y_true.reshape(-1,1) #aqui el cambio que hago es que 
        self.out = -np.mean(y_true*np.log(y_pred) + (1-y_true)*np.log(1-y_pred))
        return self.out
    
    def backward(self, y_pred, y_true):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1-epsilon)
        y_true = y_true.reshape(-1,1)

        self.grad = np.where(y_true == 1, -1 / y_pred, 1 / (1-y_pred))
        return self.grad
