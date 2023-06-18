"""
Categoriacal Cross Entropy 
-It is used to calculate loss in Multi-Class Classification
"""

import math 

softmax_output = [0.7,0.2,0.1]
target_output = [1,0,0]

loss = -(target_output[0]*math.log(softmax_output[0]) +
         target_output[1]*math.log(softmax_output[1]) +
         target_output[2]*math.log(softmax_output[2]))

print(loss)

# Below thing will also gives same output
loss = -(target_output[0]*math.log(softmax_output[0]))
print(loss)