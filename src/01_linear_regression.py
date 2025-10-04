"""
Linear Regression Implementation from Scratch
Author: Nish Patel
Dataset: NYC Taxi 2022
Date: 03/10/2025
"""

#importing neccessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Our variables
class Linear_regression_from_scratch:
    def __init__(self, L_rate = 0.005, iteration = 10000):
        self.iteration = iteration
        self.L_rate = L_rate

    def fit(self,X,Y,theta0=0):
        self.theta = np.zeros(X.shape[1])    
        self.theta0 = theta0
        m = len(Y)

        #Training iteration
        for i in range(1,self.iteration+1):
            hypo = X.dot(self.theta)+ theta0
            error = (1/(2*m))*(np.sum((hypo-Y)**2))
            grad0 = (1/m)*(np.sum((hypo-Y)))
            grad = (1/m)*((X.T.dot(hypo-Y)))
            self.theta0 -= self.L_rate*grad0
            self.theta -= self.L_rate*grad
        
        if (i%(self.iteration/10) == 0):
            print(f"iteration: {i}, error: {error}")
    def predict(self, X):
        return f"{X.dot(self.theta) + self.theta0:.2f}"
    
if __name__ == "__main__":
    #Choosing NYC Taxi 2022 dataset with 4 atrget columns randomly
    Data = pd.read_csv("Dataset/Yello Taxi.csv")
    X = Data[["trip_distance","passenger_count","tip_amount","tolls_amount"]]
    Y = Data["total_amount"]

    #Train the model
    model = Linear_regression_from_scratch()
    model.fit(X,Y)
    print(f"Model succesfully trained!!")

    #making predictions
    print(model.predict(np.array([5,5,5,5])))
