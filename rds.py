# Import Libraries
import os
import time
import glob
import json
import pymysql
import requests
import datetime
import numpy as np
import pandas as pd


DATA_PREDICTIONS = "./data/predictions/"


def push_data_to_RDS():
  def create_conn():
    """ Create connection to RDS wyth PyMySQL. 

    Returns:
        pymysql [String]: SQL Credentials.
    """
    print("\nINITIALIZED CONNECTION TO RDS...")

    dir_path = os.path.dirname(os.path.realpath(__file__))
    f = open(os.path.join(dir_path, "aws_keys"), "r")
    keys = f.read().split("\n")

    return pymysql.connect(
        host=keys[2],
        user=keys[3],
        password=keys[4],
        port=int(keys[5]))


  def get_predicted_data():
    """ Get all csv previously created after making new predictions.

    Returns:
        df [dataframe]: Return a dataframe with all csv with their trending.
    """
    print("\nIMPORTING PREDICTED DATASET(S)...")
    os.chdir(DATA_PREDICTIONS)

    df = pd.DataFrame()

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

    df = pd.DataFrame(zip_code, columns=["code_postal"])

    print("\nDF_ZIPCODE\n", df)



    # Get all CSV file
    all_csv = [i for i in glob.glob('*.csv')]

    print("\nALL_CSV\n", all_csv)
    print("\ntype - ALL_CSV\n", type(all_csv))
    
    # Convert CSV to Dataframe and merge them all in one
    for i in all_csv:
      df = pd.merge(df, pd.read_csv(i), on="code_postal", how="right")

    print("\nDF\n", df)
    print(df.shape)

    keys = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    values = [-62.5, -37.5, -17.5, -7.5, -2.5, 2.5, 7.5, 17.5, 37.5, 62.5, 87.5, 112.5, 137.5, 162.5, 187.5]

    converter = dict(zip(keys, values))

    df_converter_prediction_1 = pd.DataFrame(converter.items(),
                                             columns=['prediction_1', 
                                                      'prediction_1*'])

    df_converter_prediction_3 = pd.DataFrame(converter.items(),
                                             columns=['prediction_3', 
                                                      'prediction_3*'])

    df = pd.merge(df, df_converter_prediction_1)
    df = pd.merge(df, df_converter_prediction_3)

    del df['prediction_1']
    del df['prediction_3']

    df = df.rename(columns={'prediction_1*': 'prediction_1'})
    df = df.rename(columns={'prediction_3*': 'prediction_3'})

    print("\nDF - CAT -> %\n", df)

    # Sort dataframe ascending with column "code_postal"
    df = df.sort_values(by=['code_postal'])

    print("\nDF - SORTED & READY TO SEND TO RDS\n", df)


    # Locally saved dataframe send to RDS in "./data/data_to_rds.csv"
    df.to_csv("../data_to_rds.csv", index=False)

    return df


  def send_to_rds(df, conn):
    """ Upload a dataframe in RDS by PyMySQL.

    Args:
        df [Dataframe]: Get a dataframe to upload.
        conn [String]: SQL Statement to RDS.
    """
    print("\nSENDING DATA TO RDS...")

    cursor = conn.cursor()
    header_data = ["code_postal", "prediction_1", "prediction_3", "prix_m2_appart", "prix_m2_maison"]
    header_data = ','.join(header_data)

    set_database = "USE predimmo;"
    cursor.execute(set_database)
    conn.commit()

    remove_content_table = "DELETE FROM prediction"
    cursor.execute(remove_content_table)
    conn.commit()

    for index, row in df.iterrows():
      insert_data = []
      
      insert_data.append(str(row['code_postal']))
      insert_data.append(str(row['prediction_1']))
      insert_data.append(str(row['prediction_3']))
      insert_data.append(str(row['prix_m2_appart']))
      insert_data.append(str(row['prix_m2_maison']))

      sql = "REPLACE INTO prediction(" + header_data + ") VALUES (" + "%s,"*(len(row)-1) + "%s)"
      
      print(sql)
      print(" > ({})".format(index), insert_data)
    
      cursor.execute(sql, tuple(insert_data))
      conn.commit()
    
    print("\n(V) DONE!")


  def display_table_RDS():
    """ Get all data on AWS RDS Database and display it in terminal. That let us check the last commit on RDS Database.
    """
    conn = create_conn()
    print(" > DISPLAY DATA FROM RDS\n")

    try:
      with conn.cursor() as cursor:
        request = "SELECT * FROM predimmo.prediction"
        cursor.execute(request)
        result = cursor.fetchall()
    finally:
      conn.close()

    print(pd.DataFrame(result))


  conn = create_conn()   
  df = get_predicted_data()
  send_to_rds(df, conn)
  display_table_RDS()
