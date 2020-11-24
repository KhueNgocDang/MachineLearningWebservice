import joblib
import pandas as pd


class LinearRegressionPrediction:
    def __init__(self):
        path_to_artifacts = "../../research/"
        self.model = joblib.load(path_to_artifacts + "linear_regression")

    def preprocessing(self, input_data):
        input_data = pd.DataFrame(input_data)
        return input_data

    def predict(self, input_data):
        return self.model.predict_proba(input_data)

    def compute_prediction(self, input_data):
        try:
            input_data = self.preprocessing(input_data)
            prediction = self.predict(input_data)
        except Exception as e:
            return {"status": "Error", "message": str(e)}

        return prediction
