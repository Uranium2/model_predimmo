import datetime
import pandas as pd

from sklearn.preprocessing import LabelEncoder


DATA_DIR = "./data/"
DATA_RAW = DATA_DIR + "raw/"
DATA_TRAIN = DATA_DIR + "train/prediction_current_year/"

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
                    "surface_reelle_bati": int, 
                    "nombre_pieces_principales": int, 
                    "surface_terrain": int})


def load_df():
    """ Load a dataframe with a specific year.

    Returns:
         df [Dataframe]: Return a dataframe with all loaded dataframe in folder "data".
    """
    print("\nCREATING DATAFRAME FROM LASTEST DATASET...")
    
    print(year)
    print(type(year))

    df = pd.read_csv(DATA_RAW + "75_" + str(year - 1) + ".csv", encoding="utf-8")
    df = df[["valeur_fonciere","code_postal", "code_type_local", "surface_reelle_bati", "nombre_pieces_principales", "surface_terrain", "longitude", "latitude"]]
    
    df.reset_index(drop=True)
    
    df = df.drop_duplicates()
    
    df = df[df.code_type_local >= 1]
    df = df[df.code_type_local <= 2]

    df["valeur_fonciere"] = df["valeur_fonciere"].fillna(df["valeur_fonciere"].mean())
    df["code_postal"] = df["code_postal"].fillna(df['code_postal'].mean())
    df["code_type_local"] = df["code_type_local"].fillna(df['code_type_local'].mean())
    df["surface_reelle_bati"] = df["surface_reelle_bati"].fillna(df['surface_reelle_bati'].mean())
    df["nombre_pieces_principales"] = df["nombre_pieces_principales"].fillna(df['nombre_pieces_principales'].mean())
    df["surface_terrain"] = df["surface_terrain"].fillna(0)

    df = df.groupby(["code_postal", "code_type_local", "longitude", "latitude"])["valeur_fonciere", "surface_reelle_bati", "nombre_pieces_principales", "surface_terrain"].agg('mean').reset_index()
    
    return df


def preprocessing_current_year():
    """ Create one dataset by merging two consecutive years with percent increase or decrease (property value Y & Y+1) and add labelization for each row.

    Create:
        Y_Y+3.csv [.csv]: Create new dataset with consecutive year (ex: "2015_2018.csv").
    """
    df = load_df()

    print("\n > DATASET RAW\n", df)

    df = reset_type(df)

    df = df[["code_postal", "code_type_local", "valeur_fonciere", "surface_reelle_bati"]]
    print("\n > DATASET CURRENT YEAR => READY FOR PREDICTIONS\n", df)

    # Create new merged dataset without duplicates
    print("\nWRITTING...")

    df.to_csv(DATA_TRAIN + str(year - 1) + ".csv", index=False)
    
    print(" (+) Successfully created dataset in \'" + DATA_TRAIN + str(year - 1) + ".csv\'\n")
