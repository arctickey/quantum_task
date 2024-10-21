
from abc import ABC, abstractmethod
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import root_mean_squared_error
import pandas as pd
from typing import Union
from sklearn.base import RegressorMixin
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
import pickle


class BaseModel(ABC):
    """
    Abstract base class for any model used for prediction.
    """

    @abstractmethod
    def fit(self, train: pd.DataFrame) -> RegressorMixin:
        """Fit the model with training data."""
        pass

    @abstractmethod
    def predict(self, model: RegressorMixin, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the trained model."""
        pass

    def calculate_rmse(self, model: RegressorMixin, X: Union[pd.DataFrame, np.ndarray], y_true: Union[pd.Series, np.ndarray]) -> float:
        """
        Calculate RMSE for the given model and dataset.
        
        Parameters:
        - model: Trained model.
        - X: Input features.
        - y_true: True target values.
        
        Returns:
        - RMSE score as float.
        """
        y_pred = model.predict(X)
        return root_mean_squared_error(y_true, y_pred)

    def cross_validate(self, model: RegressorMixin, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray], cv: int = 5) -> float:
        """
        Perform cross-validation on the model and return average RMSE.
        
        Parameters:
        - model: Trained model to validate.
        - X: Input features.
        - y: Target values.
        - cv: Number of cross-validation folds.
        
        Returns:
        - Average RMSE across folds as float.
        """
        scores = -cross_val_score(model, X, y, cv=cv, scoring='neg_root_mean_squared_error')
        return scores.mean()

    def save_model(self, model: RegressorMixin, file_path: str) -> None:
        """
        Save the trained model to a pickle file.
        
        Parameters:
        - model: Trained model.
        - file_path: File path where model will be saved.
        
        Returns:
        - None
        """
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)


class QuadraticRegression(BaseModel):
    """
    Quadratic Regression model using only Feature 6 due to its correlation with the target.
    """
    
    def fit(self, train: pd.DataFrame) -> RegressorMixin:
        """
        Fit a quadratic regression model using only Feature 6.
        
        Parameters:
        - train: Training dataset with features and target.
        
        Returns:
        - Trained model as RegressorMixin.
        """
        X = train[['6']]
        y = train['target']
        poly_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
        poly_model.fit(X, y)
        return poly_model

    def predict(self, model: RegressorMixin, X: pd.DataFrame) -> np.ndarray:
        """
        Predict using the quadratic regression model.
        
        Parameters:
        - model: Trained model.
        - X: Input features for prediction.
        
        Returns:
        - Predicted values as np.ndarray.
        """
        return model.predict(X[['6']])


class KNNModel(BaseModel):
    """
    KNeighborsRegressor model using all features.
    """
    
    def fit(self, train: pd.DataFrame) -> RegressorMixin:
        """
        Fit a KNeighborsRegressor model using all features.
        
        Parameters:
        - train: Training dataset with features and target.
        
        Returns:
        - Trained model as RegressorMixin.
        """
        X = train.drop(columns=['target'])
        y = train['target']
        rf_model = KNeighborsRegressor()
        rf_model.fit(X, y)
        return rf_model

    def predict(self, model: RegressorMixin, X: pd.DataFrame) -> np.ndarray:
        """
        Predict using the KNeighborsRegressor model.
        
        Parameters:
        - model: Trained model.
        - X: Input features for prediction.
        
        Returns:
        - Predicted values as np.ndarray.
        """
        return model.predict(X)