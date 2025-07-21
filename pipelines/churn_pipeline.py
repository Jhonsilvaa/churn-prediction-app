"""Churn Prediction Pipeline Module.

This module contains the ChurnPipeline class, which is responsible for loading pre-trained machine learning models and performing data preprocessing and churn predictions data.
"""

from typing import Dict

import joblib
import pandas as pd
from scipy.special import expit


class ChurnPipeline:
    """Pipeline for churn prediction using a machine learning model.

    Loads pre-trained models and preprocessors from joblib files,
    encodes categorical features, scales numerical features, and makes predictions.

    Attributes:
        encoder (OneHotEncoder): The encoder used for categorical features.
        features (list): List of feature names.
        scaler (StandardScaler): The scaler used for numerical features.
        model (SGDClassifier): The trained model for churn prediction.
    """

    def __init__(self, features, encoder, scaler, model) -> None:
        """
        Initialize prediction pipeline components.

        Args:
           features (str): Path to the joblib file containing the list of feature names.
           encoder (str): Path to the joblib file containing the OneHotEncoder.
           scaler (str): Path to the joblib file containing the StandardScaler.
           model (str): Path to the joblib file containing the trained SGDClassifier model.
        """
        self.encoder = joblib.load(encoder)
        self.features = joblib.load(features)
        self.scaler = joblib.load(scaler)
        self.model = joblib.load(model)

    def preprocess(self, data: Dict[str, str | int | float]) -> pd.DataFrame:
        """
        Preprocess input data for model prediction.

        Args:
            data: Raw input features as dictionary.

        Returns:
            pd.DataFrame: Preprocessed DataFrame with encoded and scaled features.
        """
        df_input = pd.DataFrame([data])
        cat_cols = df_input.select_dtypes(include='object').columns
        num_cols = df_input.select_dtypes(include='number').columns
        encoded = self.encoder.transform(df_input[cat_cols].copy())
        encoded.reset_index(inplace=True)
        encoded.drop(columns='index', inplace=True)
        scaled = self.scaler.transform(df_input[num_cols].copy())
        df_scaled = pd.DataFrame(scaled, columns=num_cols)
        df_transform = pd.concat([encoded, df_scaled], axis=1)
        return df_transform

    def predict(
        self, data: Dict[str, str | int | float]
    ) -> Dict[str, int | float]:
        """
        Make age prediction from input data.

        Args:
            data: Raw input features as dictionary.

        Returns:
            dict: Dictionary with prediction (0 or 1) and churn probability.
        """
        df_transform = self.preprocess(data)
        prediction = self.model.predict(df_transform)
        scores = self.model.decision_function(df_transform)
        probas = expit(scores)
        return {'prediction': int(prediction[0]), 'proba': float(probas[0])}
