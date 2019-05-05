# Support Vector Regression

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:3].values


# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)


# Fitting SVR to the dataset
from sklearn.svm import SVR
regressor= SVR(kernel='rbf')
regressor.fit(X,y)


# Predict a new result with SVR
#y_pred=sc_y.inverse_transform(regressor.predict(sc_X.fit_transform([[6.5]])))
y_pred=sc_y.inverse_transform(regressor.predict(X))


# Visualising the SVR 
plt.scatter(X,y,color='red')
plt.plot(X,regressor.predict(X),color='blue')
plt.title('True or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()



# High resolution or better visualising the SVR
X_grid=np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color='red')
plt.plot(X_grid,regressor.predict(X_grid),color='blue')
plt.title('True or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()






