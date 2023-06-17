"""
EXPLANATION:
Neuron_Output = input*weight + bias
It's Straight Line (y = mx + c)
"""

import numpy as np
inputs = [1.2,4.3,2.4]
weights = [[2.1,3.2,1.2],
  [4.1,2.2,5.2],
  [6.1,3.2,8.2]]

biases = [3,2,7]

# With Numpy 
output_layer_neurons = np.dot(weights,inputs) + biases



"""
# With Loops 
output_layer_neurons = []
for neuron_weights,neuron_bias in zip(weights,biases):
    neuron_output = 0
    for neuron_input,neuron_weight in zip(neuron_weights,inputs):
        neuron_output+=neuron_input*neuron_weight
    neuron_output +=neuron_bias
    output_layer_neurons.append(neuron_output)

"""
print(output_layer_neurons)

