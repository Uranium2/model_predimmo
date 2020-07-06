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


DATA_DIR = "./data/processed/"
SAVE_MODEL_NAME = "_model_predimmo"


def training_classification():
  def data_preprocessing():
    # Import preprocessed datasets
    print("\nIMPORT DATASETS...")

    df1 = pd.read_csv(DATA_DIR + "2015_2016.csv", encoding="utf-8")
    df2 = pd.read_csv(DATA_DIR + "2016_2017.csv", encoding="utf-8")
    df3 = pd.read_csv(DATA_DIR + "2017_2018.csv", encoding="utf-8")
    df4 = pd.read_csv(DATA_DIR + "2018_2019.csv", encoding="utf-8")

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

    # Slice dataset and create Matrix (X) & Vector (y)
    X = df[["code_postal", "latitude", "longitude", "code_type_local", "surface_reelle_bati", "surface_terrain", "nombre_pieces_principales", "valeur_fonciere_x"]]
    X = pd.get_dummies(X)

    y = df[["categorie"]]

    print("MATRIX\n", X)
    print("VECTOR\n", y)

    # Standardize features and label
    x = X.values # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    X = pd.DataFrame(x_scaled)

    y = tf.keras.utils.to_categorical(y, 15)

    # Split dataset into train and testset
    X_train = X[:][:1096]
    X_val = X[:][1096:]

    y_train = y[:][:1096]
    y_val = y[:][1096:]

    print("X_TRAIN\n", X_train)
    print("X_VAL\n", X_val)
    print("Y_TRAIN\n", y_train)
    print(y_train.shape)
    print("Y_VAL\n", y_val)
    print(y_val.shape)

    return X_train, X_val, y_train, y_val


  def deep_learning_model(X_train, X_val, y_train, y_val):
    # HYPERPARAMETERS FOR NEURAL NETWORK (NN)
    EPOCHS = 700
    BATCH_SIZE = 64
    NEURONES_ACTIVATION = 512
    NEURONES_PREDICTION = 15
    LEARNING_RATE = 0.001
    OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    ACTIVATION = "relu"
    PREDICTION = "softmax"

    # Create NN model
    model = tf.keras.models.Sequential([
      tf.keras.layers.Dense(NEURONES_ACTIVATION, activation=ACTIVATION, name='Relu_hidden_layer_1', input_dim = X_train.shape[1]),
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


    history = model.fit(X_train,
                        y_train,
                        BATCH_SIZE,
                        EPOCHS,
                        validation_data=(X_val, y_val)
    )

    datestr = time.strftime("%Y-%m-%d")
    # timestr = time.strftime("_%H:%M:%S") # WIP - OSError: Unable to create file.


    ## Display accuracy for the different step
    print("\n#################################################################")
    print("######################   MODEL'S RESULTS   ######################")
    print("#################################################################\n")

    model.summary()

    print("\nACCURACY")
    print('> Training:', (history.history['categorical_accuracy'][len(history.history['categorical_accuracy']) - 1]) * 100, "%")
    print('> Validation:', (history.history['val_categorical_accuracy'][len(history.history['val_categorical_accuracy']) - 1]) * 100, "%")

    print("\nLOSS")
    print('> Training:', (history.history['loss'][len(history.history['loss']) - 1]))
    print('> Validation:', (history.history['val_loss'][len(history.history['val_loss']) - 1]))

    # Save the entire model to a HDF5 file
    model.save("./save/model/" + datestr + SAVE_MODEL_NAME + ".h5")

    print("\n--- Training duration: %s ---" % (str(datetime.timedelta(seconds = (time.time() - start_time)))))
    
    print("\n(+) Successfully saved MLP model \'./save/model/" + SAVE_MODEL_NAME + ".h5\'\n")

    print("#################################################################\n")

    # Get prediction
    # res = model.predict(X[0:10])

    # print("\nPREDICTIONS\n", res)

  X_train, X_val, y_train, y_val = data_preprocessing()
  deep_learning_model(X_train, X_val, y_train, y_val)