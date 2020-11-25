import csv
import pandas as pd
from django.test import TestCase

from apps.ml.stockprice_prediction.lr_predict import LinearRegressionPrediction

class MLTests(TestCase):
    def test_rf_algorithm(self):
        input_data = pd.read_csv(r"C:\Users\Khue\PycharmProjects\MachineLearningWebservice\backend\server\apps\ml\excel_aaa.csv")
        my_alg = LinearRegressionPrediction()
        response = my_alg.compute_prediction(input_data)
