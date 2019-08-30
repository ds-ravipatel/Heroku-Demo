#this is - Polynomial Linear Regression Excersize
import numpy as np
import pandas as pd
from sklearn import linear_model
from matplotlib import pyplot as plt
from sklearn import preprocessing

X=pd.read_csv('Poly_LR.csv')
y=X.Salary
X=X.drop(['Position','Salary'],axis=1)

#this is linear Regression 
l_model = linear_model.LinearRegression()

l_model.fit(X,y)

#graph to show how it is not perfect/good for our example
plt.scatter(X,y,color='r')
plt.plot(X,l_model.predict(X),color='b')
plt.show()

# use of polynomial Linear Regression
pl_model = preprocessing.PolynomialFeatures(3)
X_poly = pl_model.fit_transform(X)

l_model_new = linear_model.LinearRegression()

l_model_new.fit(X_poly,y)

#graph to show how it is perfect/good for our example
plt.scatter(X,y,color='r')
plt.plot(X,l_model_new.predict(X_poly),color='b')
plt.title('Position-Salary Mapping')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#lets store this trained model
import pickle

pickle.dump(l_model_new,open('Poly_LR_Model.pkl','wb'))

Loaded_model = pickle.load(open('Poly_LR_Model.pkl','rb'))

print(Loaded_model.predict([[1,4,16,64]]))

#below is for Polynomial Linear Regression exercize for Social Ads
#not polynomial. Its seems to be logistics regression

X1=pd.read_csv('lr_poly_social_ads.csv')
X1.describe()
X1.info()
y1 = X1.Purchased
X1 = X1.drop(['User ID'],axis = 1)
X1.replace(['Male','Female'],['1','0'], inplace = True)
