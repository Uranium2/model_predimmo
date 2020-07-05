import pandas as pd

from model import nn_classification
from preprocessing import preprocessing

if __name__ == "__main__":
    preprocessing()
    nn_classification()
