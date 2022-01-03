#Linear Regression을 통해 분포를 나타내는 직선을 찾고
#그 직선을 기준으로 데이터를 위(1), 아래(0) 등으로 분류함 

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import SGD, Adam
import os
import numpy as np

print(tf.__version__)

nowpath = os.getcwd()
print(nowpath)

loaded_data = np.loadtxt('./NeoWizard/diabetes.csv', delimiter=',')

# training data / test data 분리

seperation_rate = 0.3  # 분리 비율
test_data_num = int(len(loaded_data) * seperation_rate)

np.random.shuffle(loaded_data)

test_data = loaded_data[ 0:test_data_num ]
training_data = loaded_data[ test_data_num: ]

# training_x_data / training_t__data 생성

training_x_data = training_data[ :, 0:-1]
training_t_data = training_data[ :, [-1]]

# test_x_data / test_t__data 생성
test_x_data = test_data[ :, 0:-1]
test_t_data = test_data[ :, [-1]]

print("loaded_data.shape = ", loaded_data.shape)
print("training_x_data.shape = ", training_x_data.shape)
print("training_t_data.shape = ", training_t_data.shape)

print("test_x_data.shape = ", test_x_data.shape)
print("test_t_data.shape = ", test_t_data.shape)

model = Sequential()
sgd = SGD(learning_rate=0.01)

model.add(Dense(training_t_data.shape[1], input_shape=(training_x_data.shape[1],), activation='sigmoid'))
model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
    
from datetime import datetime

start_time = datetime.now()
hist = model.fit(training_x_data, training_t_data, epochs=500, validation_split=0.2, verbose=2)
end_time = datetime.now()
print('\nElapsed Time => ', end_time - start_time)

model.evaluate(test_x_data, test_t_data)

import matplotlib.pyplot as plt

plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()

plt.plot(hist.history['loss'], label='train loss')
plt.plot(hist.history['val_loss'], label='validation loss')

plt.legend(loc='best')

plt.show()

#accuracy levl varies in epoch=200 -> can evaluate of "epoch" 's quality   
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.grid()

plt.plot(hist.history['accuracy'], label='train accuracy')
plt.plot(hist.history['val_accuracy'], label='validation accuracy')

plt.legend(loc='best')

plt.show()