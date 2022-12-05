# -*- coding: utf-8 -*-
"""
  Download to Fashion MNIST Data
"""
from tensorflow import keras

(train_input, train_target), (test_input, test_target) =\
  keras.datasets.fashion_mnist.load_data()

"""
  Create Train-set and Test-set using train_test_split
"""
from sklearn.model_selection import train_test_split

train_scaled = train_input / 255.0
train_scaled = train_scaled.reshape(-1, 28*28)
train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size=0.2, random_state=42
)

"""
  Create Deep Neural Network model
"""
dense = keras.layers.Dense(10, activation="sigmoid", input_shape=(784,))
dense2 = keras.layers.Dense(10, activation="softmax")

model = keras.Sequential([dense, dense2])

"""
  Using summary()
"""
model.summary()

"""
  Another way to Deep Neural Network model add layers
"""
#Defalut
dense_1 = keras.layers.Dense(10, activation="sigmoid", input_shape=(784,))
dense_2 = keras.layers.Dense(10, activation="relu")
model = keras.Seuential(dense_1, dense_2)
#Solution-1
model = keras.Sequential([
    keras.layers.Dense(10, activation="sigmoid", input_shape=(784,), name="hidden"),
    keras.layers.Dense(10, activation="softmax", name="output"),
    ],  name="Fashion MNIST Model")
model.summary()
#Solution-2
model = keras.Sequential()
model.add(keras.layers.Dense(10, activation="sigmoid", input_shape=(784,)))
model.add(keras.layers.Dense(10, activation="softmax"))
model.summary()

"""
  Start to fit for Multi-layer Perceptron Model
"""
model.compile(loss="sparse_categorical_crossentropy", metrics="accuracy")
model.fit(train_scaled, train_target, epochs=5)

"""
  Flatten Class
"""
model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28, 28)))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax")) #passed input_shape, automating
model.summary()

"""
  Creating and training a new Multi-layer Perceptron Model
"""
#load data
(train_input, train_target), (test_input, test_target) =\
  keras.datasets.fashion_mnist.load_data()
#train_test_split
train_scaled = train_input / 255.0
train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size=0.2, random_state=42
)
#create Model
model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28, 28)))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))
#start to fit
model.compile(loss="sparse_categorical_crossentropy", metrics="accuracy")
model.fit(train_scaled, train_target, epochs=5)

"""
  Check the accuracy of the model
"""
model.evaluate(val_scaled, val_target)

"""
  Using Optimizer
  1. optimizers.SGD()
  2. optimizers.Adam()
  3. optimizers.Adagrad()
  4. optimizers.RMSprop()
"""
#tensorflow.keras.optimizers.SGD()
model.compile(
    optimizer="sgd",
    loss="sparse_categorical_crossentropy", 
    metrics="accuracy"
    )
model.fit(train_scaled, train_target, epochs=5)
model.evaluate(val_scaled, val_target)
#Another way
sgd = keras.optimizers.SGD()
model.compile(
    optimizer=sgd,
    loss="sparse_categorical_crossentropy", 
    metrics="accuracy"
    )
model.fit(train_scaled, train_target, epochs=5)
model.evaluate(val_scaled, val_target)

#tensorflow.keras.optimizers.Adam()
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy", 
    metrics="accuracy"
    )
model.fit(train_scaled, train_target, epochs=5)
model.evaluate(val_scaled, val_target)

#tensorflow.keras.optimizers.Adagrad()
model.compile(
    optimizer="adagrad", 
    loss="sparse_categorical_crossentropy", 
    metrics="accuracy"
    )
model.fit(train_scaled, train_target, epochs=5)
model.evaluate(val_scaled, val_target)

#tensorflow.keras.optimizers.RMSprop()
model.compile(
    optimizer="rmsprop",
    loss="sparse_categorical_crossentropy",
    metrics="accuracy"
    )
model.fit(train_scaled, train_target, epochs=5)
model.evaluate(val_scaled, val_target)