import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data() print(x_train[0])
plt.imshow(x_train[6],cmap=plt.cm.binary) plt.show()
print(y_train[6])
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1) print(x_train[0]) plt.imshow(x_train[6],cmap=plt.cm.binary) plt.show()
model = tf.keras.models.Sequential() model.add(tf.keras.layers.Flatten()) model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax)) model.compile(optimizer='Nadam', loss='sparse_categorical_crossentropy',
metrics=['accuracy'])
his=model.fit(x_train, y_train, epochs=15)
val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss)
print(val_acc)
model.save('epic_num_reader.model')
new_model = tf.keras.models.load_model('epic_num_reader.model') predictions = new_model.predict(x_test)
print(predictions)
import numpy as np
print(np.argmax(predictions[70]))
plt.imshow(x_test[70],cmap=plt.cm.binary) plt.show()