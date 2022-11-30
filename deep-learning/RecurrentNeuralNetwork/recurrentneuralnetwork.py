# -*- coding: utf-8 -*-
"""
  Using IMDB data-set
"""
from tensorflow.keras.datasets import imdb

(train_input, train_target), (test_input, test_target) = imdb.load_data(num_words=500)

print(train_input.shape, test_input.shape)
print(train_input[:10])
print(train_target[:10])

"""
  Create a validation-set
"""
from sklearn.model_selection import train_test_split

train_input, valid_input, train_target, valid_target = train_test_split(
    train_input, train_target, random_state=42, test_size=0.2
)

"""
  Analysis of train-set
"""
import numpy as np

length = np.array([len(x) for x in train_input])
print(np.mean(length), np.median(length))

"""
  Data histogram visualization
"""
import matplotlib.pyplot as plt

plt.hist(length)
plt.xlabel("length")
plt.ylabel("frequency")
plt.title("length histogram")
plt.show()

"""
  Train-set preprocessing
"""
from tensorflow.keras.preprocessing.sequence import pad_sequences

train_seq = pad_sequences(train_input, maxlen=100)
print(train_seq.shape)
print(train_seq[0])
print(train_seq[0][-10:])

"""
  One-hot encoding
"""
from tensorflow import keras

train_oh = keras.utils.to_categorical(train_seq)
print("train_oh\n",train_oh.shape)
print(train_oh[0][0][:12])
print(np.sum(train_oh[0][0]))

"""
  Valid-set preprocessing
"""
valid_seq = pad_sequences(valid_input, maxlen=100)
#one-hot encoding
valid_oh = keras.utils.to_categorical(valid_seq)
print("valid_oh\n",valid_oh.shape)
print(valid_oh[0][0][:12])
print(np.sum(valid_oh[0][0]))

"""
  Create a Recurrent neural network(RNN) model
"""
#SimpleRNN
model = keras.Sequential([
    keras.layers.SimpleRNN(8, input_shape=(100, 500)),
    keras.layers.Dense(1, activation="sigmoid")
])
model.summary()

"""
  Training a Recurrent neural network model
"""
rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4) #lr= 0.0001
model.compile(optimizer=rmsprop, loss="binary_crossentropy", metrics=["accuracy"])
checkpoint_model = keras.callbacks.ModelCheckpoint("best-simplernn-model.h5", save_best_only=True,)
early_stopping_model = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)

history = model.fit(
    train_oh, train_target, epochs=50, batch_size=64, validation_data=(valid_oh, valid_target),
    callbacks=([checkpoint_model, early_stopping_model])
)

"""
  Check training loss and validation loss with graph
"""
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend(["train", "valid"])
plt.title("Simple RNN model loss graph")
plt.show()

"""
  Using Embedding to create a embed model
"""
embed_model = keras.Sequential([
    keras.layers.Embedding(500, 16, input_length=(100)),
    keras.layers.SimpleRNN(8),
    keras.layers.Dense(1, activation="sigmoid")
])
embed_model.summary()

"""
  Training a Recurrent neural network model with Embedding
"""
rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4) #lr= 0.0001
embed_model.compile(optimizer=rmsprop, loss="binary_crossentropy", metrics=["accuracy"])
checkpoint_embed_model = keras.callbacks.ModelCheckpoint("best-embedding-model.h5", save_best_only=True,)
early_stopping_embed_model = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)

history = embed_model.fit(
    train_seq, train_target, epochs=50, batch_size=64, validation_data=(valid_seq, valid_target),
    callbacks=([checkpoint_embed_model, early_stopping_embed_model])
)

"""
  Check training loss and validation loss with graph
"""
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend(["train", "valid"])
plt.title("embed model loss graph")
plt.show()