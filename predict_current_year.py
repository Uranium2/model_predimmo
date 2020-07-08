# Import Libraries
import os
import time
import pymysql
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DATA_TRAIN = "./data/train/prediction_current_year/"
DATA_PREDICTIONS = "./data/predictions/"
SAVE_MODEL_DIR = "./save/model/"


year = datetime.date.today().year


def predict_current_year():
  def get_dataset():
    print("\nIMPORTING DATASET...")

    df = pd.read_csv(DATA_TRAIN + str(year - 1) + ".csv", encoding="utf-8")

    print(df)

    # for index, row in df.iterrows():
    #   if df["code_type_local"] == 2:
    #     print("Appartement")
    #     # df["valeur_fonciere"].mean().round()
    #   elif df["code_type_local"] == 1:
    #     print("Maison")

    exit()
    return df


  def data_preprocessing():
    

    return 


  def prediction_to_csv(X_init, y):
    y = [np.argmax(i) for i in y]

    X_init['prediction_3'] = y
    df_pred = X_init[['code_postal', 'prediction_3']].groupby(['code_postal']).mean().round()
    df_pred = df_pred.reset_index()

    df_pred = df_pred.astype({"prediction_3": int})

    print("PREDICTIONS FOR EACH DEPARTMENTS\n", df_pred)

    df_pred.to_csv(DATA_PREDICTIONS + "prediction_3years.csv", index=False)


  df = get_dataset()
  y = deep_learning_model(X)
  prediction_to_csv(X_init, y)
