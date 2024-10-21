


import pandas as pd
from sklearn.base import RegressorMixin
import pickle

def run_on_test_data(model:RegressorMixin,test: pd.DataFrame,predictions_name_to_save:str) -> None:
    """
    Function to use the fitted models and create predictions for unseen data.
    """
    preds = pd.Series(model.predict(test))
    preds.to_csv(f'./data/{predictions_name_to_save}.csv')



if __name__ == "__main__":
    test = pd.read_csv("./data/hidden_test.csv",usecols=['6'])
    with open('./models/polynomial.pkl', 'rb') as file:
        loaded_model = pickle.load(file)
    name = 'polynomial_predictions'
    run_on_test_data(loaded_model,test,name)