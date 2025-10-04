Welcome to my first-ever ML project!!
This project is a very basic implimentation of linear regression using only numpy.

## Learning
I learnt the formulas and processes through Stanford's CS229 Course taught by andrew ng 

## Libraries
1.numpy: for array, vectors and summations
2.Pandas: for Data handling
3.matplotlib.pyplot: used for data visualization to testing and analysing model

## Features
- Vectorized implementation for efficiency
- Gradient descent optimization
- Loss tracking and visualization

## Dataset
NYC Yellow Taxi 2022 Dataset
Features used: trip_distance, passenger_count, tip_amount, tolls_amount
Target: total_amount

## Usage
from linear_regression import LinearRegressionFromScratch

model = LinearRegressionFromScratch()
model.fit(X_train, y_train)
predictions = model.predict(X_test)