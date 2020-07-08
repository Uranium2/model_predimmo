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

from fake_headers import Headers


DATA_PREDICTIONS = "./data/prediction/"

header = Headers(headers=True)


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
    # print("\nAWS_KEYS\n", keys)

    return pymysql.connect(
        host=keys[2],
        user=keys[3],
        password=keys[4],
        port=int(keys[5]))


  def get_predicted_data():
    print("\nIMPORTING PREDICTED DATASET(S)...")
    os.chdir(DATA_PREDICTIONS)

    # Get all CSV file
    all_csv = [i for i in glob.glob('*.csv')]
    
    # Convert CSV to Dataframe and concatenate them all in one
    df = pd.concat([pd.read_csv(f) for f in all_csv])

    print(df)
    print(df.shape)

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
    conn = create_conn()
    print("PRINT DATA FROM RDS")

    try:
      with conn.cursor() as cursor:
        request = "SELECT * FROM predimmo.prediction"
        print("REQUEST\n", request)
        cursor.execute(request)
        result = cursor.fetchall()
    finally:
      conn.close()

    print("\nTABLE: prediction\n", pd.DataFrame(result))


  conn = create_conn()   
  df = get_predicted_data()
  send_to_rds(df, conn)
  display_table_RDS()
