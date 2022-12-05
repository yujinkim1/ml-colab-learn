# -*- coding: utf-8 -*-
"""
  Download to Fashion MNIST Data
"""
from tensorflow import keras
(train_input, train_target), (test_input, test_target) =\
  keras.datasets.fashion_mnist.load_data()

"""
  Create train-set and test-set using `train_test_split()`
"""
from sklearn.model_selection import train_test_split
train_scaled = train_input / 255.0
train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size=0.2, random_state=42
)
#checkout
print(train_scaled.shape)
print(val_scaled.shape)

"""
  Create Deep Neural Network model(DNN)
"""
#create a model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax"),
])
model.compile(loss="sparse_categorical_crossentropy", metrics="accuracy")
model.fit(train_scaled, train_target, epochs=5)

"""
  Create model to using model_fn()
"""
def model_fn(a_layer=None):
  model = keras.Sequential()
  model.add(keras.layers.Flatten(input_shape=(28, 28)))
  model.add(keras.layers.Dense(100, activation="relu"))
  if a_layer:
    model.add(a_layer)
  model.add(keras.layers.Dense(10, activation="softmax"))
  return model
#request functions
model = model_fn()
#using summary method
model.summary()

"""
  Model compile and output to history
"""
model.compile(loss="sparse_categorical_crossentropy", metrics="accuracy")
#create history
history = model.fit(train_scaled, train_target, epochs=5, verbose=0)
#checkout
print(history.history.keys())

"""
  Visualize the loss of a model
"""
import matplotlib.pyplot as plt
plt.plot(history.history["loss"])
plt.title("Model loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()

plt.plot(history.history["accuracy"])
plt.title("Model accuracy")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.show()

"""
  Increasing epochs value
"""
#increase from 5 epoch to 20
model = model_fn()
model.compile(loss="sparse_categorical_crossentropy", metrics="accuracy")
history = model.fit(train_scaled, train_target, epochs=20, verbose=0)

#checkout
plt.plot(history.history["loss"])
plt.title("loss when increased from 5 epoch to 20")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()

plt.plot(history.history["accuracy"])
plt.title("accuracy when increased from 5 epoch to 20")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.show()

"""
  Validation loss
"""
model = model_fn()
model.compile(loss="sparse_categorical_crossentropy", metrics="accuracy")
history = model.fit(
    train_scaled, train_target, epochs=20, verbose=0,
    validation_data=(val_scaled, val_target) #forward to tuple
  )

print(history.history.keys())

#checkout
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("check validation loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend(["train", "val"])
plt.show()

"""
  Mitigating overfitting
"""
#adam 옵티마이저가 에포크에 따른 학습률의 크기 조정에 유리
model = model_fn()
#use adam optimizer
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics="accuracy")
history = model.fit(
    train_scaled, train_target, epochs=20, verbose=0,
    validation_data=(val_scaled, val_target)
  )

#checkout
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("validation loss with optimizer")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend(["train", "val"])
plt.show()

"""
  Dropout
"""
model = model_fn(keras.layers.Dropout(0.3)) #30%
model.summary()

#graph of results with dropout applied
model = model_fn()
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics="accuracy")
history = model.fit(
    train_scaled, train_target, epochs=20, verbose=0,
    validation_data=(val_scaled, val_target)
  )

#visualize 
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("dropout graph")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend(["train", "val"])
plt.show()

"""
  Saving my Best model files
"""
model = model_fn(keras.layers.Dropout(0.3))
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics="accuracy")
history = model.fit(
    train_scaled, train_target, epochs=10, verbose=0,
    validation_data=(val_scaled, val_target)
  )

#use .save_weights()
model.save_weights("model-weight.h5")
#use .save()
model.save("model-whole.h5")
"""
checkout saved model file
!ls -al *.h5
"""

"""
  Load to my Best model files
"""
#load_weights()
model = model_fn(keras.layers.Dropout(0.3))
model.load_weights("model-weight.h5")

import numpy as np
val_labels = np.argmax(model.predict(val_scaled), axis=1)
print(np.mean(val_labels == val_target))

#checkout
print(val_labels[:20])
print(val_target[:20])

#load_model()
model = keras.models.load_model("model-whole.h5")

model.evaluate(val_scaled, val_target)

"""
  Callback
"""
#create model
model = model_fn(keras.layers.Dropout(0.3))
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics="accuracy"
)

#create a checkPoint
checkPoint = keras.callbacks.ModelCheckpoint(
    "best-model.h5", save_best_only=True
)

model.fit(
    train_scaled, train_target, epochs=20, verbose=0,
    validation_data=(val_scaled, val_target),
    callbacks=[checkPoint]
)

#predict
model = keras.models.load_model("best-model.h5")
model.evaluate(val_scaled, val_target)

"""
  Early stopping point
"""
#create a model
model = model_fn(keras.layers.Dropout(0.3))
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics="accuracy"
)

#checkPoint
checkPoint = keras.callbacks.ModelCheckpoint(
    "best-model.h5", save_best_only=True
)

#earlyStopping
earlyStopping = keras.callbacks.EarlyStopping(
    patience=2,
    restore_best_weights=True
)

model.fit(
    train_scaled, train_target, epochs=20, verbose=0,
    validation_data=(val_scaled, val_target),
    callbacks=[checkPoint]
)

#predict
model = keras.models.load_model("best-model.h5")
model.evaluate(val_scaled, val_target)

#stopped epochs value
print(earlyStopping.stopped_epoch)