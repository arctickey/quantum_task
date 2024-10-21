## Project Overview
This task involves building and evaluating machine learning models to predict a target variable using a given dataset. The project includes data exploration, model training, and prediction on unseen data.


## Setup Instructions
Clone the repository:
git clone https://github.com/arctickey/quantum_task.git
cd quantum_task

## Create a virtual environment:
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

## Install the required libraries:
pip install -r requirements.txt

## Project Components
### Data
Data is not stored in github due to size reasons. In order to run the repo, plesase put data in that folder.

data/train.csv: Training dataset.

data/hidden_test.csv: Test dataset.

data/polynomial_predictions.csv: Dataset with predictions for the test set.

### Models
models/polynomial.pkl: Pickled polynomial regression model.

models/knn.pkl: Pickled KNeighborsRegressor model.

### Scripts
eda.py: Script for performing exploratory data analysis (EDA) on the training data.

train.py: Script for training models and performing cross-validation.

predict.py: Script for making predictions on the test data using the trained models.

models.py: Contains the model classes and utility functions for training and prediction.

## Note on model selection
The EDA has shown a strong quadaratic relation beetween feature 6 and the target on the training data, 
while rest of the features has shown to be relatively random. This is why in the predict script the
Polynomial Regression of degree 2 is used, with only feature 6. Its provide almost ideally correct output, with little
complexity of the model.
For th sake of completness the KNN model was trained on the whole dataset to serve as a baseline.

## Usage
In order to run the scripts head to the /src/task_2 folder and execute the given commands.
### Exploratory Data Analysis (EDA)
Run the EDA script to analyze the training data:

python eda.py

### Training Models
Train the models and perform cross-validation:

python train.py

This script will:

Train a quadratic regression model using Feature 6.
Train a KNN model using all features.
Perform cross-validation and print RMSE for both models.
Save the trained models to the models folder.

### Making Predictions
Use the trained models to make predictions on the test data:

python predict.py

This script will:

Load the test data from data/hidden_test.csv.
Use the trained models to make predictions.
Save the predictions to the data folder.

## Requirements
Ensure you have the following libraries installed (listed in requirements.txt):

- pandas
- scikit-learn
- seaborn
