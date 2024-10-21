


import pandas as pd
from src.task_2.models import QuadraticRegression, KNNModel
import logging

logger = logging.getLogger()

def train(train: pd.DataFrame) -> None:
    """
    Function to run cross-validation for both Quadratic Regression (Feature 6)
    and KNN model using all features. Prints RMSE for both models and save
    models to /models folder.
    """

    quad_model = QuadraticRegression()
    poly_fitted = quad_model.fit(train)
    X_feature_6 = train[['6']]
    y = train['target']
    poly_cv_rmse = quad_model.cross_validate(poly_fitted, X_feature_6, y)
    logger.info(f"Polynomial Regression (Feature 6) Cross-Validated RMSE: {poly_cv_rmse:.4f}")
    quad_model.save_model(poly_fitted,'./models/polynomial.pkl')

    rf_model = KNNModel()
    rf_fitted = rf_model.fit(train)
    X_all_features = train.drop(columns=["target"])
    rf_cv_rmse = rf_model.cross_validate(rf_fitted, X_all_features, y)
    logger.info(f"KNN  Cross-Validated RMSE: {rf_cv_rmse:.4f}")
    quad_model.save_model(rf_fitted,'./models/knn.pkl')


if __name__ == "__main__":
    train_set = pd.read_csv("./data/train.csv")
    train(train_set)