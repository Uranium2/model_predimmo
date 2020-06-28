import pandas as pd
import datetime
from sklearn.preprocessing import LabelEncoder


year = datetime.date.today().year


def reset_type(df):
    return df.astype({"valeur_fonciere_x": int, 
                    "valeur_fonciere_y": int, 
                    "code_postal": int, 
                    "code_type_local": int,
                    "surface_reelle_bati": int, 
                    "nombre_pieces_principales": int, 
                    "surface_terrain": int})

def load_df(year_index):
    df = pd.read_csv("./data/75_{}.csv".format(year_index), encoding="utf-8")
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

def preprocessing():
    for i in range(2016, year):
        for y in range(i + 1, year):
            print(i, y)
            df_i = load_df(i)
            df_y = load_df(y)

            big_df = pd.merge(df_i, df_y, how="inner",  on=["code_postal", "code_type_local", "surface_reelle_bati", "nombre_pieces_principales", "surface_terrain", "longitude", "latitude"])
            # big_df = big_df.rename(columns={"valeur_fonciere_x": "valeur_fonciere_{}".format(i), "valeur_fonciere_y": "valeur_fonciere_{}".format(y)})
            big_df.reset_index(drop=True)
            print(big_df)
            big_df = reset_type(big_df)
            # 100 * (y - i)/ i
            big_df['pourcentage'] = 100 * (big_df['valeur_fonciere_y'] - big_df['valeur_fonciere_x']) / big_df['valeur_fonciere_x']
            big_df = big_df[big_df.pourcentage < 200]
            big_df = big_df[big_df.pourcentage > -75]

            big_df = big_df[big_df.valeur_fonciere_y > 5000]
            big_df = big_df[big_df.valeur_fonciere_x > 5000]
            
            bins = [float(i) for i in range(-75, -10, 25)]
            bins = bins + [float(i) for i in range(-10, 10, 5)]
            bins = bins + [float(i) for i in range(10, 225, 25)]

            bins = list(dict.fromkeys(bins))

            group_names = [str(i) for i in range(0, len(bins) - 1)]
            label_quality = LabelEncoder()

            big_df['categorie'] = pd.cut(big_df['pourcentage'], bins = bins, labels = group_names)

            big_df['categorie'] = label_quality.fit_transform(big_df['categorie'])

            print(big_df['categorie'].value_counts())

        
            print("writting")
            big_df.to_csv("./data/{}_{}.csv".format(i, y), index=False)
            print("done writting")
