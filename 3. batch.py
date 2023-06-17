
import numpy as np
batches = [
    [1.2,4.3,2.4,3.5],
    [5.2,2.3,4.4,9.4],
    [2.2,8.3,1.4,4.1]]
weights1 = [[2.1,3.2,1.2,2.1],
  [4.1,2.2,5.2,3.1],
  [6.1,3.2,8.2,3.2]
  ]

biases1 = [3,2,7]

weights2 = [[1.2,2.5,7.4],
  [8.1,6.2,2.2],
  [3.1,7.2,4.2]
  ]

biases2 = [1,4,2]

# With Numpy 
output_layer_1 = np.dot(batches, np.array(weights1).T) + biases1
output_layer_2 = np.dot(output_layer_1, np.array(weights2).T) + biases2



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
print(output_layer_2)

