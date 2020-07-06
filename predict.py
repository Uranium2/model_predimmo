# Import Libraries
import os
import time
import pymysql
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
SAVE_MODEL_DIR = "./save/model/"
SAVE_MODEL_NAME = "_model_predimmo"

### ToDo ###
# 1 - Pull RDS.
# 2 - Create df.
# 3 - Data-Preprocessing.
# 4 - Use Model + Predict => y_pred.
# 5 - Push df (SQL) => RDS.

def predict_classification():
  def create_conn():
    """ Create connection to RDS wyth PyMySQL. 

    Returns:
        pymysql [String]: SQL Credentials.
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    f = open(os.path.join(dir_path, "aws_keys.txt"), "r")
    keys = f.read().split("\n")
    # print("\nAWS_KEYS\n", keys)

    return pymysql.connect(
        host=keys[2],
        user=keys[3],
        password=keys[4],
        port=int(keys[5]))


  def get_data_RDS():
    # 1 - Pull RDS.
    
    conn = create_conn()

    print("\n(V) Successfully connected to RDS: Predimmo.data_django.")

    try:
      with conn.cursor() as cursor:
        request = "SELECT * FROM predimmo.data_django"
        # print("REQUEST\n", request)
        cursor.execute(request)
        result = cursor.fetchall()
        # print("\nRESULT\n", result)
    finally:
      conn.close()

    return result

  def data_preprocessing():
    # 2 - Create df.
    # 3 - Data-Preprocessing
    
    # result = get_data_RDS()
    # print("(V) Successfully pulled all data from RDS: Predimmo.data_django.")

    # df = pd.DataFrame(result, columns=["date_mutation", "code_postal", "valeur_fonciere", "code_type_local", "surface_reelle_bati", "nombre_pieces_principales","surface_terrain","longitude","latitude","message"])

    # df.to_csv('./leboncoin.csv', index=False)

    # print("\n(+) Successfully create \'./leboncoin.csv\' in EC2 - model_predimmo.\n")
    
    # Analyse leboncoin dataset
    df = pd.read_csv("leboncoin.csv", encoding="utf-8")
    
    print("\nDATAFRAME RAW\n", df)

    df = df.drop(columns=['date_mutation', 'message'])
    
    print("\nDATAFRAME FOR PREDICTION\n", df)

    # print(type(df['code_postal']))
    # exit()
    # Change dataset's types
    df = df.astype({"code_postal": str, 
                    "valeur_fonciere": int, 
                    "code_type_local": int,
                    "surface_reelle_bati": int, 
                    "nombre_pieces_principales": int, 
                    "surface_terrain": int})

    print("\nDATAFRAME NEW TYPES\n", df)

    # Slice dataset and create Matrix (X) & Vector (y)
    X = df[["code_postal", "latitude", "longitude", "code_type_local", "surface_reelle_bati", "surface_terrain", "nombre_pieces_principales", "valeur_fonciere"]]
    X = pd.get_dummies(X)

    print("MATRIX\n", X)

    # Standardize features and label
    x = X.values # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    X = pd.DataFrame(x_scaled)

    print("X - STD\n", X)

    return X


  # def deep_learning_model(X_train, X_val, y_train, y_val):
  #   # 4 - Use Model + Predict => y_pred.
  #   # HYPERPARAMETERS FOR NEURAL NETWORK (NN)
  #   EPOCHS = 700
  #   BATCH_SIZE = 64
  #   NEURONES_ACTIVATION = 512
  #   NEURONES_PREDICTION = 15
  #   LEARNING_RATE = 0.001
  #   OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
  #   ACTIVATION = "relu"
  #   PREDICTION = "softmax"

  #   # Create NN model
  #   model = tf.keras.models.Sequential([
  #     tf.keras.layers.Dense(NEURONES_ACTIVATION, activation=ACTIVATION, name='Relu_hidden_layer_1', input_dim = X_train.shape[1]),
  #     tf.keras.layers.Dense(NEURONES_ACTIVATION, activation=ACTIVATION, name='Relu_hidden_layer_2'),
  #     tf.keras.layers.Dense(NEURONES_ACTIVATION, activation=ACTIVATION, name='Relu_hidden_layer_3'),
  #     tf.keras.layers.Dense(NEURONES_ACTIVATION, activation=ACTIVATION, name='Relu_hidden_layer_4'),
  #     tf.keras.layers.Dense(NEURONES_PREDICTION, activation=PREDICTION, name='Softmax_predictions_output')
  #   ])

  #   model.summary()

  #   # Train NN & evaluate training: Accuracy & Loss
  #   # Start timer to get learning time
  #   start_time = time.time()

  #   # Specify the training configuration (optimizer, loss & metrics)
  #   model.compile(optimizer=OPTIMIZER,
  #                 loss='categorical_crossentropy',
  #                 metrics=['categorical_accuracy']
  #   )


  #   history = model.fit(X_train,
  #                       y_train,
  #                       BATCH_SIZE,
  #                       EPOCHS,
  #                       validation_data=(X_val, y_val)
  #   )

  #   datestr = time.strftime("%Y-%m-%d")
  #   # timestr = time.strftime("_%H:%M:%S") # WIP - OSError: Unable to create file.


  #   ## Display accuracy for the different step
  #   print("\n#################################################################")
  #   print("######################   MODEL'S RESULTS   ######################")
  #   print("#################################################################\n")

  #   model.summary()

  #   print("\nACCURACY")
  #   print('> Training:', (history.history['categorical_accuracy'][len(history.history['categorical_accuracy']) - 1]) * 100, "%")
  #   print('> Validation:', (history.history['val_categorical_accuracy'][len(history.history['val_categorical_accuracy']) - 1]) * 100, "%")

  #   print("\nLOSS")
  #   print('> Training:', (history.history['loss'][len(history.history['loss']) - 1]))
  #   print('> Validation:', (history.history['val_loss'][len(history.history['val_loss']) - 1]))

  #   # Save the entire model to a HDF5 file
  #   model.save("./save/model/" + datestr + SAVE_MODEL_NAME + ".h5")

  #   print("\n--- Training duration: %s ---" % (str(datetime.timedelta(seconds = (time.time() - start_time)))))
    
  #   print("\n(+) Successfully saved MLP model \'./save/model/" + SAVE_MODEL_NAME + ".h5\'\n")

  #   print("#################################################################\n")

  #   # Get prediction
  #   # res = model.predict(X[0:10])

  #   # print("\nPREDICTIONS\n", res)


  # def push_predicted_data_RDS():
  #   # 5 - Push df (SQL) => RDS.

  data_preprocessing()
