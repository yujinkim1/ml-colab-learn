# -*- coding: utf-8 -*-
"""
    Using IMDB data-set
"""
from tensorflow.keras.datasets import imdb
from sklearn.model_selection import train_test_split
#load to imdb data
(train_input, train_target), (test_input, test_target) = imdb.load_data(
    num_words= 500
)
#data-set split
train_input, valid_input, train_target, valid_target = train_test_split(
    train_input, train_target, random_state= 42, test_size= 0.2
)

#checkout
print(train_input.shape)
print(valid_input.shape)

"""
    Dataset Preprocessing
"""
from tensorflow.keras.preprocessing.sequence import pad_sequences

train_seq = pad_sequences(train_input, maxlen=100)
valid_seq = pad_sequences(valid_input, maxlen=100)

#checkout
print(train_seq.shape)
print(valid_seq.shape)

"""
    Create LSTM model
"""
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Embedding(500, 16, input_length=100),
    keras.layers.LSTM(8),
    keras.layers.Dense(1, activation="sigmoid")
])

model.summary()

"""
    Training to LSTM model
"""
rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4)
model.compile(optimizer=rmsprop, loss="binary_crossentropy", metrics=["accuracy"])
checkpoint_model = keras.callbacks.ModelCheckpoint("best-lstm-model.h5", save_best_only=True)
early_stopping_model = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)

history = model.fit(
    train_seq, train_target, epochs=100, batch_size=64,
    validation_data=(valid_seq, valid_target),
    callbacks=[checkpoint_model, early_stopping_model]
    )

"""
    Visualization LSTM model loss graph
"""
import matplotlib.pyplot as plt

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.xlabel("loss")
plt.ylabel("epochs")
plt.title("LSTM model loss graph")
plt.legend(["train", "valid"])
plt.show()

"""
    Using Dropouts for recurrent layer
"""
dropout_model = keras.Sequential([
    keras.layers.Embedding(500, 16, input_length=100),
    keras.layers.LSTM(8, dropout=0.3),
    keras.layers.Dense(1, activation="sigmoid")
])

dropout_model.summary()

"""
    Training to LSTM model with Dropout
"""
rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4)
dropout_model.compile(optimizer=rmsprop, loss="binary_crossentropy", metrics=["accuracy"])
checkpoint_dropout_model = keras.callbacks.ModelCheckpoint("best-dropout-model.h5", save_best_only=True)
early_stopping_dropout_model = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)

history = dropout_model.fit(
    train_seq, train_target, epochs=100, batch_size=64,
    validation_data=(valid_seq, valid_target),
    callbacks=[checkpoint_dropout_model, early_stopping_dropout_model]
    )

"""
    Visualization LSTM model loss graph with Dropout
"""
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.xlabel("loss")
plt.ylabel("epochs")
plt.title("LSTM model loss graph with Dropout")
plt.legend(["train", "valid"])
plt.show()

"""
    Two recurrent layers Connect with recurrent neural network model
"""
nested_model = keras.Sequential([
    keras.layers.Embedding(500, 16, input_length=100),
    keras.layers.LSTM(8, dropout=0.3, return_sequences=True),
    keras.layers.LSTM(8, dropout=0.3),
    keras.layers.Dense(1, activation="sigmoid")
]) 

nested_model.summary()

"""
    Training to Two recurrent layers Connect with recurrent neural network model
"""
rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4)
nested_model.compile(optimizer=rmsprop, loss="binary_crossentropy", metrics=["accuracy"])
checkpoint_nested_model = keras.callbacks.ModelCheckpoint("best-nested-model.h5", save_best_only=True)
early_stopping_nested_model = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)

history = nested_model.fit(
    train_seq, train_target, epochs=100, batch_size=64,
    validation_data=(valid_seq, valid_target),
    callbacks=[checkpoint_nested_model, early_stopping_nested_model]
    )

"""
    Visualization Two recurrent layers Connect with RNN model loss graph
"""
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.xlabel("loss")
plt.ylabel("epochs")
plt.title("Two recurrent layers Connected RNN model loss graph")
plt.legend(["train", "valid"])
plt.show()

"""
    Create Gated Recurrent Unit Neural Network model
"""
gru_model = keras.Sequential([
  keras.layers.Embedding(500, 16, input_length=100),
  keras.layers.GRU(8),
  keras.layers.Dense(1, activation="sigmoid")     
])

gru_model.summary()

"""
    Training to GRU model
"""
rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4)
gru_model.compile(optimizer=rmsprop, loss="binary_crossentropy", metrics=["accuracy"])
checkpoint_gru_model = keras.callbacks.ModelCheckpoint("best-gru-model.h5", save_best_only=True)
early_stopping_gru_model = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)

history = gru_model.fit(
    train_seq, train_target, epochs=100, batch_size=64,
    validation_data=(valid_seq, valid_target),
    callbacks=[checkpoint_gru_model, early_stopping_gru_model]
    )

"""
    Visualization GRU model loss graph
"""
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.xlabel("loss")
plt.ylabel("epochs")
plt.title("GRU model loss graph")
plt.legend(["train", "valid"])
plt.show()
