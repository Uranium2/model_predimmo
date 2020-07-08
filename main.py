import pandas as pd

from preprocessing_1year import preprocessing_1year
from preprocessing_3years import preprocessing_3years
from preprocessing_current_year import preprocessing_current_year
from model_1year import training_classification_1year
from model_3years import training_classification_3years
from predict_1year import predict_classification_1year
from predict_3years import predict_classification_3years
from predict_current_year import predict_current_year
from rds import push_data_to_RDS

if __name__ == "__main__":
    # preprocessing_1year()
    # preprocessing_3years()
    # preprocessing_current_year()
    # training_classification_1year()
    # training_classification_3years()
    # predict_classification_1year()
    # predict_classification_3years()
    predict_current_year()
    # push_data_to_RDS()
