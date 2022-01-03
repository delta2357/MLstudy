import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import SGD, Adam

import numpy as np
import matplotlib.pyplot as plt

#inserts X_data array and t_data array 
#wx + w'x' + w''x'' + b = t
x_data = np.array([ 
[4], [2], [5], [1], [6], [7], [1], [14], [2], [3]]) 

t_data = np.array([62, 56, 65, 53, 68, 71, 53, 92, 56, 59])

#build model with 1 dense layer and print summary
model=Sequential()
model.add(Dense(1, input_shape=(1,), activation ='linear'))

sgd = SGD(learning_rate=1e-2)
model.compile(optimizer=sgd, loss='mse')
model.summary()

#fit model with x_data and t_data
hist = model.fit(x_data, t_data, epochs=500)

test_data = [ [4], [2], [1]]
ret_val = [ 62, 56, 53 ] 

prediction_val = model.predict(np.array(test_data))

print(prediction_val)
print('-===========-')
print(ret_val)

#visualize loss trend with epoch stage
plt.title("Loss Trend")
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()

plt.plot(hist.history['loss'], label='train loss')
plt.legend(loc='best')

plt.show()
