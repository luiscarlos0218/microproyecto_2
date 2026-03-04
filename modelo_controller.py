import numpy as np
import os.path as osp
import pandas as pd
from DataPreprocessing import DataPreprocessing
import os
import joblib

class ModelController:

    def __init__(self):
        print("ModelController.__init__ ->")
        # Asegura en una variable la ruta de los modelos      

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(BASE_DIR, "obj", "modelo.joblib")

        self.model = joblib.load(model_path)


    def predict(self, text):
        print("ModelController.predict ->")

        preprocesador = DataPreprocessing()
        X= preprocesador.text_preprocess(text)
        y_pred = self.model.predict([X])
        y_prob= self.model.predict_proba([X])

        return X, y_pred, y_prob
