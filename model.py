import numpy as np
import pandas as pd

import sklearn as sk
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

def nn():
  ### ---------- DATA PREPROCESSING ---------- ###

  # Get preprocessed datasets
  df1 = pd.read_csv("./data/2015_2016.csv", encoding="utf-8")
  df2 = pd.read_csv("./data/2016_2017.csv", encoding="utf-8")
  df3 = pd.read_csv("./data/2017_2018.csv", encoding="utf-8")
  df4 = pd.read_csv("./data/2018_2019.csv", encoding="utf-8")

  # Concatenate all datasets in one
  df = pd.concat([df1, df2])
  df = pd.concat([df, df3])
  df = pd.concat([df, df4])

  # Change types
  df = df.astype({"valeur_fonciere_x": int, 
                      "valeur_fonciere_y": int, 
                      "code_postal": str, 
                      "code_type_local": int,
                      "surface_reelle_bati": int, 
                      "nombre_pieces_principales": int, 
                      "surface_terrain": int})

  # Create Matrice & Vector
  X = df[["code_postal", "latitude", "longitude", "code_type_local", "surface_reelle_bati", "surface_terrain", "nombre_pieces_principales", "valeur_fonciere_x"]]
  X = pd.get_dummies(X)
  y = df[["categorie"]]

  # Split dataset into training, validation and test set
  # X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(x,
  #                                                                       y, 
  #                                                                       test_size=0.3)
  # X_train, X_val, y_train, y_val = sk.model_selection.train_test_split(X_train, 
  #                                                                     y_train, 
  #                                                                     test_size=0.2)

  # print("X_train\n", X_train, "\n")
  # print("X_val\n", X_val, "\n")
  # print("X_test\n", X_test, "\n")

  # print("y_train\n", y_train, "\n")
  # print("y_val\n", y_val, "\n")
  # print("y_test\n", y_test, "\n")

  # scaler = StandardScaler()

  # X_train_sc = scaler.fit_transform(X_train)
  # X_val_sc = scaler.transform(X_val)
  # X_test_sc = scaler.transform(X_test)

  # y_train_sc = scaler.fit_transform(y_train)
  # y_val_sc = scaler.transform(y_val)
  # y_test_sc = scaler.transform(y_test)

  # print("X_train_sc\n", X_train_sc)
  # print("\nX_val_sc\n", X_val_sc)
  # print("\nX_test_sc\n", X_test_sc)

  # print("y_train_sc\n", y_train_sc)
  # print("\ny_val_sc\n", y_val_sc)
  # print("\ny_test_sc\n", y_test_sc)

  y = tf.keras.utils.to_categorical(y, 15)
  x = X.values # returns a numpy array

  min_max_scaler = preprocessing.MinMaxScaler()
  x_scaled = min_max_scaler.fit_transform(x)

  X = pd.DataFrame(x_scaled)
  # print(X)
  # print(y)


  ### ---------- DEEP LEARNING ---------- ###

  # HYPERPARAMETERS
  EPOCHS = 1000
  BATCH_SIZE = 10000
  NEURONES_ACTIVATION = 256
  NEURONES_PREDICTION = 15
  BATCH_SIZE = 64
  LEARNING_RATE = 0.001
  OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
  ACTIVATION = "relu"
  PREDICTION = "softmax"

  # Create NN model
  model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(NEURONES_ACTIVATION, activation=ACTIVATION, name='Relu_hidden_layer_1', input_dim = X.shape[1]),
    tf.keras.layers.Dense(NEURONES_ACTIVATION, activation=ACTIVATION, name='Relu_hidden_layer_2'),
    tf.keras.layers.Dense(NEURONES_ACTIVATION, activation=ACTIVATION, name='Relu_hidden_layer_3'),
    tf.keras.layers.Dense(NEURONES_ACTIVATION, activation=ACTIVATION, name='Relu_hidden_layer_4'),
    tf.keras.layers.Dense(NEURONES_PREDICTION, activation=PREDICTION, name='Softmax_predictions_output')
  ])

  model.summary()

  model.compile(optimizer=OPTIMIZER,
                loss='categorical_crossentropy',
                metrics=['categorical_accuracy']
  )

  hist = model.fit(X,
                  y,
                  BATCH_SIZE,
                  EPOCHS
  )

  # Get prediction
  res = model.predict(X[0:10])

  print("\nPREDICTIONS\n", res)
