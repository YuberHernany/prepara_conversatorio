# net with sepal feature vs net with petal feature (versicolors detectors)

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X, y = iris.data, iris.target
y = (y==1).astype("int") # labels for versicolor detector

def train_test_val(X, y):
    X_full, X_test, y_full, y_test = train_test_split(X, y)
    X_valid, X_train = X_full[:22], X_full[22:]
    y_valid, y_train = y_full[:22], y_full[22:]
    return X_train, X_test, X_valid, y_train, y_test, y_valid

X_sepal, X_petal = X[:, :2], X[:, 2:]

Xsep_train, Xsep_test, Xsep_val, ysep_train, ysep_test, ysep_val = train_test_val(X_sepal, y)
Xpet_train, Xpet_test, Xpet_val, ypet_train, ypet_test, ypet_val = train_test_val(X_petal, y)

#models
def build_net(n_hidden):
    model = keras.models.Sequential()
    model.add(keras.layers.Input(shape=(1,2))) # inptut layer
    for _ in range(n_hidden): # hidden layers
        model.add(keras.layers.Dense(2, activation="relu"))
    model.add(keras.layers.Dense(1, activation="sigmoid")) # output layer
    return model

sepal_ann, petal_ann = build_net(3), build_net(3)

# print(sepal_ann.summary())

def compile_model(model):
    model.compile(loss="mse",
                    optimizer="sgd",
                    metrics=["accuracy"])

compile_model(sepal_ann)
compile_model(petal_ann)

checkpoint_cb_sep = keras.callbacks.ModelCheckpoint("sepal_ann.h5",
                                                    save_best_only=True)
early_stopping_sep = keras.callbacks.EarlyStopping(patience=5,
                                                    restore_best_weights=True)
checkpoint_cb_pet = keras.callbacks.ModelCheckpoint("petal_ann.h5",
                                                    save_best_only=True)
early_stopping_pet = keras.callbacks.EarlyStopping(patience=5,
                                                    restore_best_weights=True)

sepal_ann.fit(Xsep_train, ysep_train, epochs=100,
                validation_data=(Xsep_val, ysep_val),
                callbacks=[checkpoint_cb_sep, early_stopping_sep])
petal_ann.fit(Xpet_train, ypet_train, epochs=100,
                validation_data=(Xpet_val, ypet_val),
                callbacks=[checkpoint_cb_pet, early_stopping_pet])

print('\n')
print("sepal_ann score: ")
print(sepal_ann.evaluate(Xsep_test, ysep_test))
print("petal_ann score: ")
print(petal_ann.evaluate(Xpet_test, ypet_test))

# estoy obteniendo conclusiones contraintuitivas con los gráficos obtenidos en sequential_iris.ipynb