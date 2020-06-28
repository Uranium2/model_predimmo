import pandas as pd

from model import nn_classification
from preprocessing import preprocessing

if __name__ == "__main__":
    preprocessing()
    nn_classification()
    # df = pd.read_csv("./data/2016_2017.csv", encoding="utf-8")
    # df = df.sort_values(by=['pourcentage'], ascending=False)
    # print(df)
