import numpy as np
import pandas as pd
import sklearn as sk
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

df1 = pd.read_csv("./data/2016_2017.csv", encoding="utf-8")
df2 = pd.read_csv("./data/2017_2018.csv", encoding="utf-8")
df3 = pd.read_csv("./data/2018_2019.csv", encoding="utf-8")

df = pd.concat([df1, df2])
df = pd.concat([df, df3])


df = df.astype({"valeur_fonciere_x": int, 
                    "valeur_fonciere_y": int, 
                    "code_postal": str, 
                    "code_type_local": int,
                    "surface_reelle_bati": int, 
                    "nombre_pieces_principales": int, 
                    "surface_terrain": int})

X = df[["code_postal", "code_type_local", "longitude", "latitude", "valeur_fonciere_x", "surface_reelle_bati", "nombre_pieces_principales", "surface_terrain"]]
X = pd.get_dummies(X)
Y = df[["categorie"]]

Y = tf.keras.utils.to_categorical(Y, 15)
x = X.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
X = pd.DataFrame(x_scaled)
print(X)
print(Y)


model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(256, activation='relu'),
  tf.keras.layers.Dense(256, activation='relu'),
  tf.keras.layers.Dense(256, activation='relu'),
  tf.keras.layers.Dense(256, activation='relu'),
  tf.keras.layers.Dense(15, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])

hist = model.fit(X, Y, epochs=5000, batch_size=10000)

res = model.predict(X[0:10])

print(res)