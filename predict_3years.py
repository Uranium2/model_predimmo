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


DATA_TEST = "./data/test/"
DATA_PREDICTIONS = "./data/predictions/"
SAVE_MODEL_DIR = "./save/model/"
MODEL_ARTIFACT = "2020-07-08_model_predimmo_3years.h5"
MODEL_NAME = SAVE_MODEL_DIR + MODEL_ARTIFACT


def predict_classification_3years():
  def create_conn():
    """ Create connection to RDS wyth PyMySQL. 

    Returns:
        pymysql [String]: SQL Credentials.
    """
    print("\nINITIALIZED CONNECTION TO RDS...")

    dir_path = os.path.dirname(os.path.realpath(__file__))
    f = open(os.path.join(dir_path, "aws_keys"), "r")
    keys = f.read().split("\n")
    # print("\nAWS_KEYS\n", keys)

    return pymysql.connect(
        host=keys[2],
        user=keys[3],
        password=keys[4],
        port=int(keys[5]))


  def get_data_RDS():
    """ Get data from AWS RDS database.

    Returns:
        result [list]: Return SQL request in list.
    """
    conn = create_conn()
    print("IMPORTING DATASET FROM RDS => PREDICTIONS...")
    print(" (V) Successfully connected to RDS: Predimmo.data_django.")

    try:
      with conn.cursor() as cursor:
        request = "SELECT * FROM predimmo.data_django WHERE code_postal BETWEEN 75001 AND 75020"
        print("REQUEST\n", request)
        cursor.execute(request)
        result = cursor.fetchall()
        # print("\nRESULT\n", result)
    finally:
      conn.close()

    return result


  def data_preprocessing():
    """ Data-preprocess dataset to correctly using it with deep learning model.

    Returns:
        X_init, X [Dataframe]: Initial matric dataframe and another normalized.
    """
    result = get_data_RDS()

    print(" (V) Successfully pulled all data from RDS: Predimmo.data_django.")

    df = pd.DataFrame(result, columns=["date_mutation", 
                                      "code_postal", 
                                      "valeur_fonciere", 
                                      "code_type_local", 
                                      "surface_reelle_bati", 
                                      "nombre_pieces_principales", 
                                      "surface_terrain", 
                                      "longitude", 
                                      "latitude", 
                                      "message"])

    df.to_csv(DATA_TEST + "leboncoin.csv", index=False)
    
    print("\nSTART DATA-PREPROCESSSING...")
    print(" (+) Successfully create \'./leboncoin.csv\' in EC2 - model_predimmo.")
    
    # Analyse leboncoin dataset
    df = pd.read_csv(DATA_TEST + "leboncoin.csv", encoding="utf-8")
    
    print("\nDATAFRAME RAW\n", df.head())
    print(df.shape)

    df = df.drop(columns=['date_mutation', 'message'])
    
    # Change dataset's types
    df = df.astype({"code_postal": str, 
                    "valeur_fonciere": int, 
                    "code_type_local": int,
                    "surface_reelle_bati": int, 
                    "nombre_pieces_principales": int, 
                    "surface_terrain": int})

    print("\nDATAFRAME WITH CORRECT COLUMNS AND TYPES\n", df.head())

    # Slice dataset and create Matrix (X) & Vector (y)
    X_init = df[["code_postal", 
            "latitude", 
            "longitude", 
            "code_type_local", 
            "surface_reelle_bati", 
            "surface_terrain", 
            "nombre_pieces_principales",
            "valeur_fonciere"]]

    X = pd.get_dummies(X_init)

    print("\nMATRIX - ONE HOT ENCODED\n", X.head())

    # Standardize features and label
    x = X.values # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    X = pd.DataFrame(x_scaled)

    print("\nMATRIX - STANDARDIZED\n", X.head())

    return X_init, X


  def deep_learning_model(X):
    """ Script for making new predictions on new data and return it.

    Args:
        X [dataframe]): Matrix dataframe previoulsy data-preprocessed.

    Returns:
        y [dataframe]: Return predictions.
    """
    print("\nRELOAD DEEP LEARNING MODEL...\n")

    # HYPERPARAMETERS FOR NEURAL NETWORK (NN)
    LEARNING_RATE = 0.001
    OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    LOSS = 'categorical_crossentropy'
    METRICS = ['categorical_accuracy']

    # Load NN model and recreate the exact same model, including its weights and the optimizer
    test_model = tf.keras.models.load_model(MODEL_NAME)

    # Show the model architecture
    test_model.summary()

    print("\nGENERATE NEW PREDICTIONS => NEW DATASET...")
    # Start timer to get learning time
    start_time = time.time()

    # We need to compile model to made new predictions        
    test_model.compile(optimizer=OPTIMIZER,
                      loss=LOSS,
                      metrics=METRICS
    )

    y = test_model.predict(X)
    print("\nNEW PREDICTIONS (10st)\n", y[:10])
    print('\nPredictions\'s shape', y.shape)

    datestr = time.strftime("%Y-%m-%d")
    # timestr = time.strftime("_%H:%M:%S") # WIP - OSError: Unable to create file.

    ## Display accuracy for the different step
    print("\n#################################################################")
    print("######################   MODEL'S RESULTS   ######################")
    print("#################################################################\n")

    test_model.summary()

    print("\nACCURACY")
    print("> Training      ~ XX,XX % (from training script).")
    print("> Validation    ~ XX,XX % (from training script).")
    # print("> Test           ", accuracy_rate * 100, "%")

    print("\nLOSS")
    print("> Training      ~ X,XXXXXXX...")
    print("> Validation    ~ X,XXXXXXX...")
    # print("> Test           ", y[0])

    print("\n--- Training duration: %s ---" % (str(datetime.timedelta(seconds = (time.time() - start_time)))))
    
    print("\n#################################################################\n")

    return y


  def prediction_to_csv(X_init, y):
    """ Convert dataframe to csv before sending it to RDS.

    Args:
        X_init [dataframe]: Initial matrix for preprocessed dataframe.
        y [dataframe]: New predictions.

    Create:
        .../prediction_Xyears.csv [CSV]: Create 1 prediction dataframe in 1 CSV.
    """
    y = [np.argmax(i) for i in y]

    X_init['prediction_3'] = y
    df_pred = X_init[['code_postal', 'prediction_3']].groupby(['code_postal']).mean().round()
    df_pred = df_pred.reset_index()

    df_pred = df_pred.astype({"prediction_3": int})

    print("PREDICTIONS FOR EACH DEPARTMENTS\n", df_pred)

    df_pred.to_csv(DATA_PREDICTIONS + "prediction_3years.csv", index=False)
    print(" (+) Successfully created dataset in \'" + DATA_PREDICTIONS + "prediction_3years.csv\'\n")


  X_init, X = data_preprocessing()
  y = deep_learning_model(X)
  prediction_to_csv(X_init, y)
