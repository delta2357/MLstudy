# 60000 * 28 * 28 / label * 10 (0~9)
import tensorflow as tf
import numpy as np 
from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt

(x_train, t_train), (x_test, t_test) = fashion_mnist.load_data()

'''
plt.figure(figsize=(6,6))
for index in range(25):
    plt.subplot(5, 5, index+1)
    plt.imshow(x_train[index], cmap='gray')
    plt.axis('off')

plt.show()
'''

x_train = (x_train - 0.0) / (255.0 - 0.0)
x_test = (x_test - 0.0) / (255.0 - 0.0)

#skip One-hot Encoding
#t_train = tf.keras.utils.to_categorical(t_train, num_classes=10)
#t_test = tf.keras.utils.to_categorical(t_test, num_classes=10)

model = tf.keras.Sequential()

model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(100, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

hist = model.fit(x_train, t_train, epochs=30, validation_split=0.3)

model.evaluate(x_test, t_test)

plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()
plt.plot(hist.history['loss'], label='train loss')
plt.plot(hist.history['val_loss'], label='validation loss')
plt.legend(loc='best')
plt.show()

plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.grid()
plt.plot(hist.history['accuracy'], label='train accuracy')
plt.plot(hist.history['val_accuracy'], label='validation accuracy')
plt.legend(loc='best')
plt.show()