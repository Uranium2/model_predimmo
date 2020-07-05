# Import Libraries
import os
import time
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn as sk
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential


SAVE_MODEL_NAME = "_model_predimmo"


def nn_classification():
  ### ---------- DATA PREPROCESSING ---------- ###

  # Import preprocessed datasets
  print("\nIMPORT DATASETS...")

  df1 = pd.read_csv("./data/2015_2016.csv", encoding="utf-8")
  df2 = pd.read_csv("./data/2016_2017.csv", encoding="utf-8")
  df3 = pd.read_csv("./data/2017_2018.csv", encoding="utf-8")
  df4 = pd.read_csv("./data/2018_2019.csv", encoding="utf-8")

  print("\n> DONE!")

  # Concatenate all datasets in one
  print("\nCONCATENATE ALL DATASET IN ONE...")

  df = pd.concat([df1, df2])
  df = pd.concat([df, df3])
  df = pd.concat([df, df4])
  
  print("\n> DONE!")

  # Change dataset's types
  df = df.astype({"valeur_fonciere_x": int, 
                      "valeur_fonciere_y": int, 
                      "code_postal": str, 
                      "code_type_local": int,
                      "surface_reelle_bati": int, 
                      "nombre_pieces_principales": int, 
                      "surface_terrain": int})

  df_train = df[:][:1096]
  df_test = df[:][1096:]
  print(df_train)
  print(df_test)

  # Slice dataset and create Matrix (X) & Vector (y)
  X = df_train[["code_postal", "latitude", "longitude", "code_type_local", "surface_reelle_bati", "surface_terrain", "nombre_pieces_principales", "valeur_fonciere_x"]]
  X = pd.get_dummies(X)

  y = df_train[["categorie"]]

  print("MATRIX\n", X)
  print("VECTOR\n", y)

  # Split dataset into train and testset
  # X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(X,
  #                                                                       y, 
  #                                                                       test_size=0.2)

  # print("X_train\n", X_train, "\n")
  # print("X_test\n", X_test, "\n")

  # print("y_train\n", y_train, "\n")
  # print("y_test\n", y_test, "\n")

  # scaler = StandardScaler()

  # X_train_sc = scaler.fit_transform(X_train)
  # X_test_sc = scaler.transform(X_test)

  # y_train_sc = scaler.fit_transform(y_train)
  # y_test_sc = scaler.transform(y_test)

  # print("X_train_sc\n", X_train_sc)
  # print("\nX_test_sc\n", X_test_sc)

  # print("y_train_sc\n", y_train_sc)
  # print("\ny_test_sc\n", y_test_sc)

  # Standardize the input features
  x = X.values # returns a numpy array
  min_max_scaler = preprocessing.MinMaxScaler()
  x_scaled = min_max_scaler.fit_transform(x)
  X = pd.DataFrame(x_scaled)

  y = tf.keras.utils.to_categorical(y, 15)
  # print(X)
  # print(y)


  ### ---------- DEEP LEARNING ---------- ###

  # HYPERPARAMETERS FOR NEURAL NETWORK (NN)
  EPOCHS = 700
  BATCH_SIZE = 512
  NEURONES_ACTIVATION = 512
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

  # Train NN & evaluate training: Accuracy & Loss
  # Start timer to get learning time
  start_time = time.time()

  # Specify the training configuration (optimizer, loss & metrics)
  model.compile(optimizer=OPTIMIZER,
                loss='categorical_crossentropy',
                metrics=['categorical_accuracy']
  )


  history = model.fit(X,
                      y,
                      BATCH_SIZE,
                      EPOCHS
  )

  datestr = time.strftime("%Y-%m-%d")
  # timestr = time.strftime("_%H:%M:%S") # OSError: Unable to create file


  ## Display accuracy for the different step
  print("\n#################################################################")
  print("######################   MODEL'S RESULTS   ######################")
  print("#################################################################\n")

  model.summary()

  print("\nACCURACY")
  print('> Training:', (history.history['categorical_accuracy'][len(history.history['categorical_accuracy']) - 1]) * 100, "%")

  print("\nLOSS")
  print('> Training:', (history.history['loss'][len(history.history['loss']) - 1]))

  # Save the entire model to a HDF5 file
  model.save("./save/model/" + datestr + SAVE_MODEL_NAME + ".h5")

  print("\n--- Training duration: %s ---" % (str(datetime.timedelta(seconds = (time.time() - start_time)))))
  
  print("\n(+) Successfully saved MLP model \'./save/model/" + SAVE_MODEL_NAME + ".h5\'\n")

  print("#################################################################\n")

  # Get prediction
  res = model.predict(X[0:10])

  # print("\nPREDICTIONS\n", res)
