# -*- coding: utf-8 -*-
"""
  Download to Fashion MNIST Data
"""
from tensorflow import keras

(train_input, train_target), (test_input, test_target) =\
  keras.datasets.fashion_mnist.load_data()
print(train_input.shape, train_target.shape)

"""
  The label to which each sample belong
"""
print([train_target[index] for index in range(10)])

"""
  Check the number of images saved by label
"""
import numpy as np

print(np.unique(train_target, return_counts=True))

"""
  Visualize MNIST data sets
"""
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 10, figsize=(10, 10))

for index in range(10):
  axs[index].imshow(train_input[index], cmap="gray_r")
  axs[index].axis("off")
plt.show()

"""
  Convert a two-dimensional array to a one-dimensional array
"""
train_scaled = train_input / 255.0
train_scaled = train_scaled.reshape(-1, 28*28)

#Checkout
print(train_scaled.shape)

"""
  Classification using logistic regression and check a cross-validate score
"""
from sklearn.model_selection import cross_validate
from sklearn.linear_model import SGDClassifier

sc = SGDClassifier(loss="log", max_iter=5, random_state=42)
scores = cross_validate(sc, train_scaled, train_target, n_jobs=-1)

#Checkout
print(np.mean(scores["test_score"]))

"""
  Increasing the iterations of SGDClassifier
"""
#increase max_iter=10
sc = SGDClassifier(loss="log", max_iter=10, random_state=42)
scores = cross_validate(sc, train_scaled, train_target, n_jobs=-1)

#Checkout
print(np.mean(scores["test_score"]))

#increase max_iter=20
sc = SGDClassifier(loss="log", max_iter=20, random_state=42)
scores = cross_validate(sc, train_scaled, train_target, n_jobs=-1)

#Checkout
print(np.mean(scores["test_score"]))

"""
  Create Train-set and Test-set using train_test_split
"""
from sklearn.model_selection import train_test_split

train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size=0.2, random_state=42
)

print(train_scaled.shape, train_target.shape)
print(val_scaled.shape, val_target.shape)

"""
  Create dense class
"""
dense = keras.layers.Dense(10, activation="softmax", input_shape=(784,))

"""
  Create keras model
"""
model = keras.Sequential(dense)

"""
  Set a loss function and metrics
"""
model.compile(loss="sparse_categorical_crossentropy", metrics="accuracy")

"""
  Start to fit
"""
model.fit(train_scaled, train_target, epochs=5)

"""
  Model performance evaluation
"""
model.evaluate(val_scaled, val_target)