from sklearn.datasets import fetch_california_housing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
import seaborn as sns


def lasso_ridge():
    df = fetch_california_housing(as_frame=True)
    dataset = pd.DataFrame(df.data)
    dataset.columns = df.feature_names
    dataset["Price"]=df.target
    y = dataset['Price']
    x = dataset.drop('Price', axis=1 )
    #Linear Regression
    lin_regressor = LinearRegression()
    mse = cross_val_score(lin_regressor,x,y,scoring='neg_mean_squared_error',cv=5)
    mean_mse = np.mean(mse)
    print(mean_mse)
    #Ridge Regression
    ridge = Ridge()
    parameters = {'alpha': [1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}
    ridge_regressor = GridSearchCV(ridge,parameters,scoring='neg_mean_squared_error',cv=5)
    ridge_regressor.fit(x,y)
    print(ridge_regressor.best_params_)
    print(ridge_regressor.best_score_)
    #Lasso Regression
    lasso = Lasso()
    parameters = {'alpha': [ 1e-15, 1e-10, 1e-8, 1e-3, 1e-2, 1, 5, 10, 20, 30, 35, 40, 45, 50, 55, 100 ]}
    lasso_regressor = GridSearchCV(lasso,parameters,scoring='neg_mean_squared_error',cv=5)
    lasso_regressor.fit(x,y)
    print(lasso_regressor.best_score_)
    print(lasso_regressor.best_params_)
    #Training the models
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)
    prediction_lasso = lasso_regressor.predict(x_test)
    prediction_ridge = ridge_regressor.predict(x_test)
    sns.displot(y_test-prediction_lasso)
    plt.show()
    sns.displot(y_test-prediction_ridge)
    plt.show()

if __name__ == '__main__':
 lasso_ridge()
