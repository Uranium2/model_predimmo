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


def reset_type(df):
    """ Get dataframe and return it with specific type for some columns.

    Args:
        df [Dataframe]: Get dataframe.

    Returns:
        df [Dataframe]: Return dataframe with new type for some columns. 
    """
    return df.astype({"valeur_fonciere": int, 
                    "code_postal": int, 
                    "code_type_local": int,
                    "surface_reelle_bati": int})


def predict_current_year():
  def get_dataset():
    """ Get last dataset and split it in the two type local (Home & appartment).

    Returns:
        df_home, df_appart [Dataframe]: predicted dataset for homes and appartments.
    """
    print("\nIMPORTING DATASET => PREDICT CURRENT YEAR...")

    df = pd.read_csv(DATA_TRAIN + str(year - 1) + ".csv", encoding="utf-8")

    print(df)

    # Initiliaze empty dataframes
    df_home = pd.DataFrame(columns=['code_postal', 
                                    'code_type_local', 
                                    'surface_reelle_bati', 
                                    'valeur_fonciere'])

    df_appart = pd.DataFrame(columns=['code_postal', 
                                    'code_type_local', 
                                    'surface_reelle_bati', 
                                    'valeur_fonciere'])

    # Split Home and Appartment in each dataframe
    print("\nSPLITING HOMES AND APPARTMENTS IN SEPARATE DATAFRAME...\n > Total rows: {}".format(df.shape[0]))

    for index, row in df.iterrows():
      if row["code_type_local"] == 1:
        df_home = reset_type(df_home.append(row))

      elif row["code_type_local"] == 2:
        df_appart = reset_type(df_appart.append(row)) 

    print('\nDF_HOME\n', df_home)
    print('\nDF_APPART\n', df_appart)

    return df_home, df_appart


  def data_preprocessing(df_home, df_appart):
    """ Data preprocess to "predict it" for home and appartments.

    Args:
        df_home [Dataframe]: Dataframe with only homes.
        df_appart [Dataframe]: Dataframe with only appartments.

    Returns:
        df_home, df_appart [Dataframe]: data-preprocessed datasets => Prediction (homes and appartments).
    """
    print("\nGROUPING BY ZIP_CODE...")

    df_home = df_home.groupby('code_postal', as_index=False)['valeur_fonciere'].mean()
    df_appart = df_appart.groupby('code_postal', as_index=False)['valeur_fonciere', 'surface_reelle_bati'].mean()

    # Calculate Appartment's price / m²
    df_appart['valeur_fonciere'] = df_appart['valeur_fonciere'] / df_appart['surface_reelle_bati']

    df_appart = df_appart.drop('surface_reelle_bati', axis=1)

    # Change types
    df_home = df_home.astype({"code_postal": int, 
                              "valeur_fonciere": int})

    df_appart = df_appart.astype({"code_postal": int, 
                                  "valeur_fonciere": int})

    # Replace missing value in df_home
    zip_code = [75001, 
                75002, 
                75003, 
                75004, 
                75005, 
                75006, 
                75007, 
                75008, 
                75009, 
                75010, 
                75011, 
                75012, 
                75013, 
                75014, 
                75015, 
                75016, 
                75017, 
                75018, 
                75019, 
                75020]

    df_zip_code = pd.DataFrame(list(zip(zip_code, zip_code)), columns=["code_postal", "x"])

    print("\nADDING MISSING ZIP CODE AND FILL WITH -1 WHEN NO DATA...")

    df_home = df_zip_code.merge(df_home, on='code_postal', how='left')

    del  df_home['x']

    df_home["valeur_fonciere"] = df_home["valeur_fonciere"].fillna(-1)
    df_home = df_home.astype({"valeur_fonciere": int})


    return df_home, df_appart


  def prediction_to_csv(df_home, df_appart):
    """ Convert 2 dataframes to 2 CSV before sending it to RDS.

    Args:
        df_home [Dataframe]: Dataframe with only homes with predictions.
        df_appart [Dataframe]: Dataframe with only appartments with predictions.
    
    Create:
        .../prix_type.csv [CSV]: Create 2 predictions dataframes in 2 CSV.
    """
    print("\nCREATING DATASETS PREDICTION...")

    df_home.to_csv(DATA_PREDICTIONS + "prix_maison.csv", index=False)
    df_appart.to_csv(DATA_PREDICTIONS + "prix_m2_appart.csv", index=False)

    print('DF_HOME\n', df_home)
    print('\nDF_APPART\n', df_appart)

    print("\n (+) Successfully created dataset in \'" + DATA_PREDICTIONS + "prix_maison.csv\'")
    print(" (+) Successfully created dataset in \'" + DATA_PREDICTIONS + "prix_m2_appart.csv\'\n")

  df_home, df_appart = get_dataset()
  df_home, df_appart = data_preprocessing(df_home, df_appart)
  prediction_to_csv(df_home, df_appart)
