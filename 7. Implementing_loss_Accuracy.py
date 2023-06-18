import math
import numpy as np
softmax_output = np.array([[0.7,0.2,0.1]
                  ,[0.5,0.1,0.4]
                    ,[0.02,0.9,0.08]])
"""
Classes :
0 - Cat
1 - Dog
2 - Human
"""

target_output = [0,1,1] # One Hot Encoding-> Cat,Dog,Dog

"""
# print(softmax_output[[0,1,2],target_output]) # First Bracket is for Rows and Second is for Columns
neg_log =(-np.log(softmax_output[
        range(len(softmax_output)),target_output]))

avg_loss = np.mean(neg_log)
print(avg_loss)

"""

# But We can have Problem if we get log(0) because it will be infinity
# So we will add small value to softmax_output
# We clip the value between 1e-7 to 1-1e-7


clipped_pred_output = np.clip(softmax_output,1e-7,1-1e-7)
neg_log =(-np.log(clipped_pred_output[
        range(len(softmax_output)),target_output]))

avg_loss = np.mean(neg_log)
print(f"Loss : {avg_loss}")


predictions = np.argmax(softmax_output,axis = 1)
correct_outputs = 0
for i in range(len(predictions)):
    if predictions[i] == target_output[i]:
        correct_outputs+=1
accuracy = correct_outputs / len(predictions)
print(f"Accuracy : {accuracy}")