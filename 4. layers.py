import numpy as np

np.random.seed(0)
X = np.array([
    [1.2,4.3,2.4,3.5],
    [5.2,2.3,4.4,9.4],
    [2.2,8.3,1.4,4.1]])

class LayerDense :
    def __init__(self,n_inputs,n_neurons) -> None:
        self.weights = 0.1 * np.random.randn(n_inputs,n_neurons) # n_inputs x n_neurons (4xn_neurons)
        self.biases = np.zeros((1,n_neurons)) # 1 x n_neurons
    
    def forward(self,inputs):
        self.outputs = np.dot(inputs,self.weights) +self.biases  
 
print(X.shape)
l1 = LayerDense(X.shape[1],10)
l2 = LayerDense(10,2)

l1.forward(X)
print(l1.outputs)

l2.forward(l1.outputs)
print(l2.outputs)
