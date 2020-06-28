import pandas as pd
import seaborn as sns

from model import nn
from preprocessing import preprocessing

if __name__ == "__main__":
    preprocessing()
    nn()
    # df = pd.read_csv("./data/2016_2017.csv", encoding="utf-8")
    # df = df.sort_values(by=['pourcentage'], ascending=False)
    # print(df)
