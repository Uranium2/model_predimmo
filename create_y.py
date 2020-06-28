import pandas as pd


df = pd.read_csv("./data/big_df.csv", encoding="utf-8")

# df['y'] = df.apply(lambda row: row.valeur_fonciere_2016 - row.valeur_fonciere_2017, axis = 1) 

df['y'] = (df['valeur_fonciere_2016'] / df['valeur_fonciere_2019']) * 100
print(df)