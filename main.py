import pandas as pd

from preprocessing_1year import preprocessing_1year
from preprocessing_3years import preprocessing_3years
from model_1year import training_classification_1year
from model_3years import training_classification_3years
from predict_1year import predict_classification_1year
from predict_3years import predict_classification_3years
from rds import push_data_to_RDS

if __name__ == "__main__":
    # preprocessing_1year()
    # preprocessing_3years()       
    # training_classification_1year()
    # training_classification_3years()
    predict_classification_1year()
    predict_classification_3years()
    # push_data_to_RDS()
