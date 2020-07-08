import datetime
import pandas as pd

from sklearn.preprocessing import LabelEncoder


DATA_DIR = "./data/"
DATA_RAW = DATA_DIR + "raw/"
DATA_TRAIN = DATA_DIR + "train/prediction_1year/"

year = datetime.date.today().year


def reset_type(df):
    """ Get dataframe and return it with specific type for some columns.

    Args:
        df [Dataframe]: Get dataframe.

    Returns:
        df [Dataframe]: Return dataframe with new type for some columns. 
    """
    return df.astype({"valeur_fonciere_x": int, 
                    "valeur_fonciere_y": int, 
                    "code_postal": int, 
                    "code_type_local": int,
                    "surface_reelle_bati": int, 
                    "nombre_pieces_principales": int, 
                    "surface_terrain": int})


def load_df(year_index):
    """ Load a dataframe with a specific year.

    Args:
        year_index [Int]: Get year index to load specific dataset.

    Returns:
        big_df [Dataframe]: Return a dataframe with all loaded dataframe in folder "data".
    """
    df = pd.read_csv(DATA_RAW + "75_{}.csv".format(year_index), encoding="utf-8")
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

    big_df = df.groupby(["code_postal", "code_type_local", "longitude", "latitude"])["valeur_fonciere", "surface_reelle_bati", "nombre_pieces_principales", "surface_terrain"].agg('mean').reset_index()
    return big_df


def preprocessing_1year():
    """ Create one dataset by merging two consecutive years with percent increase or decrease (property value Y & Y+1) and add labelization for each row.

    Create:
        Y_Y+1.csv [.csv]: Create new dataset with consecutive year (ex: "2016_2017.csv").
    """
    for i in range(2015, year):
        for y in range(i + 1, year):
            print("\n > DATASET:", i, y)
            df_i = load_df(i)
            df_y = load_df(y)

            # Merge two datasets by "code_postal", "code_type_local", "surface_reelle_bati", "nombre_pieces_principales", "surface_terrain", "longitude", "latitude"
            big_df = pd.merge(df_i, df_y, how="inner",  on=["code_postal", "code_type_local", "surface_reelle_bati", "nombre_pieces_principales", "surface_terrain", "longitude", "latitude"])
            # big_df = big_df.rename(columns={"valeur_fonciere_x": "valeur_fonciere_{}".format(i), "valeur_fonciere_y": "valeur_fonciere_{}".format(y)})
            big_df.reset_index(drop=True)
           
            print(big_df)
            
            big_df = reset_type(big_df)

            # Calculate percent
            big_df['pourcentage'] = 100 * (big_df['valeur_fonciere_y'] - big_df['valeur_fonciere_x']) / big_df['valeur_fonciere_x']

            # Remove outliers
            big_df = big_df[big_df.pourcentage < 200]
            big_df = big_df[big_df.pourcentage > -75]
            big_df = big_df[big_df.valeur_fonciere_y > 5000]
            big_df = big_df[big_df.valeur_fonciere_x > 5000]

            # Create categories' ranges          
            bins = [float(i) for i in range(-75, -10, 25)]
            bins = bins + [float(i) for i in range(-10, 10, 5)]
            bins = bins + [float(i) for i in range(10, 225, 25)]

            bins = list(dict.fromkeys(bins))

            # Create categories' name
            group_names = [str(i) for i in range(0, len(bins) - 1)]
            
            label_quality = LabelEncoder()

            big_df['categorie'] = pd.cut(big_df['pourcentage'], bins=bins, labels=group_names)
            big_df['categorie'] = label_quality.fit_transform(big_df['categorie'])

            print("Categories Distribution in dataset\n", big_df['categorie'].value_counts())
   
            # Create new merged dataset without duplicates
            print("\nWRITTING...")
            big_df.to_csv(DATA_TRAIN + "{}_{}.csv".format(i, y), index=False)
            print(" (+) Successfully created dataset in \'" + DATA_TRAIN + "{}_{}.csv".format(i, y) + "\'\n")
