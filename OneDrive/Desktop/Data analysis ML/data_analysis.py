import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("dataset.csv")

 
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)


# Linear Regression Visualization

plt.scatter(X, y, color='red')
plt.plot(X, lin_reg.predict(X), color='blue')
plt.title('Linear Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show() 


lin_model_pred = lin_reg.predict([[6.5]])
lin_model_pred


# Polynomial model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Create polynomial features
poly_reg = PolynomialFeatures(degree=9)
X_poly = poly_reg.fit_transform(X)

# Train polynomial regression model
lin_reg_9 = LinearRegression()
lin_reg_9.fit(X_poly, y)

print(lin_reg_9) # Linear regresssion with 1 degree poly 


plt.plot(X, lin_reg.predict(X), color='blue')
plt.title('Linear Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')

plt.show() 

lin_model_pred = lin_reg.predict([[6.5]])
lin_model_pred

# polynomial model

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures()
X_poly = poly_reg.fit_transform(X)

poly_reg.fit(X_poly, y)
lin_reg_9 = LinearRegression()
lin_reg_9.fit(X_poly, y)

print(lin_reg) # linear regression with 1 degree poly
print(poly_reg) # poly with 9 degree
print(lin_reg_9) # linear model with 9 degree



plt.scatter(X, y, color='red')
plt.plot(X, lin_reg_9.predict(poly_reg.fit_transform(X)), color='blue')
plt.plot(X, lin_reg.predict(X), color='blue')
plt.title('Linear Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

poly_model_pred = lin_reg_9.predict((poly_reg.fit_transform([[6.5]])))
poly_model_pred