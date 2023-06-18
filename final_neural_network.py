"""
What Activation Function do? 
Activation Function is used to introduce Non-Linearity in the Neural Network
input --> Neuron((summation -> Activation Function)-->Output)
"""

import numpy as np
import math
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()

# X = np.array([
#     [1.2,4.3,2.4,3.5],  #-->Batch-1
#     [5.2,2.3,4.4,9.4],
#     [2.2,8.3,1.4,4.1]])

class LayerDense :
    """
    Fully Connected Layer
    """
    def __init__(self,n_inputs,n_neurons) -> None:
        self.weights = 0.1 * np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros((1,n_neurons))
    
    def forward(self,inputs):
        self.outputs = np.dot(inputs,self.weights) +self.biases
 
class ActivationRelu:
    """
    ReLU Activation Function
    - Used in Hidden Layers
    - Output is Positive or Zero
    - Used to introduce Non-Linearity
    """
    def forward(self,inputs):
        self.outputs = np.maximum(0,inputs)

class ActivationSoftMax:
    """
    SoftMax Activation Function
    - Used for Multi-Class Classification
    - Used in Output Layer
    - Output is in Probability
    """
    def __init__(self) -> None:
        self.outputs = []
    def forward(self,X):
        """
        Why we have substracted max of row?
        Because np.exp does not support larger values like 1000.(Overflow Issue) 
        So to make all values smaller we just substract max value.(It will make all exp values between 0 to 1)
        and keep dims means shape will be as it is.
        """
        exponentialValues = np.exp(X - np.max(X,axis=1, keepdims=True))
        probabilities = exponentialValues / np.sum(exponentialValues, axis=1, keepdims=True)
        self.outputs = probabilities

class Loss:
    def calculate(self,output,y): #y is Actual Value, output is Predicted Value
        sample_losses = self.forward(output,y)
        data_loss = np.mean(sample_losses)
        return data_loss
    
class Loss_CategoricalCrossEntropy(Loss):
    def forward(self,y_pred,y_true):
        samples = len(y_pred)
        y_pred_clip = np.clip(y_pred,1e-7,1-1e-7)
        # For Categorical-Value
        if len(y_true.shape) ==1:
            correct_confidence= (y_pred_clip[
                                        range(samples),
                                        y_true])
            
        # For One-Hot-Encoded
        elif len(y_true.shape) == 2:
            correct_confidence = np.sum(y_pred_clip*y_true,axis=1)

        negative_loss_likelihood = -np.log(correct_confidence)
        return negative_loss_likelihood
    

X,y = spiral_data(samples=100,classes=3)

# print(X.shape)
l1 = LayerDense(X.shape[1],3)
activation1 = ActivationRelu()

l2 = LayerDense(3,3) #inputs,Outputs
activation2 = ActivationSoftMax()

l1.forward(X)
activation1.forward(l1.outputs)

l2.forward(activation1.outputs)
activation2.forward(l2.outputs)
print(activation2.outputs[:5])

loss_function = Loss_CategoricalCrossEntropy()
loss = loss_function.calculate(activation2.outputs,y)

print(f"Loss : {loss}")
