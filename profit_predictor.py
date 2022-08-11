
# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the dataset
dataset = pd.read_csv("50_Startups.csv")
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:, 4].values


#Encoding Categorical Data
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
column_trans = make_column_transformer((OneHotEncoder(), [3]), remainder='passthrough')
X = column_trans.fit_transform(X)


#Splitting the dataset into the Training set and Test set 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

#Fitting Multiple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Prediciting the Test set results
y_pred = regressor.predict(X_test)