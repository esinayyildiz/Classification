from sklearn import datasets
from sklearn import model_selection
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt


boston = datasets.load_boston()
x = boston.data;
y=boston.target;
xtest,xtrain,ytest,ytrain= model_selection.train_test_split(x,y);




class_1 = Ridge(alpha=0.0001)
class_1.fit(xtrain, ytrain)
Ridge (alpha=0.001,  fit_intercept=True)
       
       
y_pred = class_1.predict(xtest)
print("test score value is  :",class_1.score(xtest,ytest))


plt.scatter(ytest,y_pred,color="black", marker="*")
plt.show()


poly = PolynomialFeatures(2)
poly.fit(xtrain)
xtrain = poly.transform(xtrain)
xtest = poly.transform(xtest)

class_2 = Ridge(alpha=200000)
class_2.fit(xtrain, ytrain)
Ridge (alpha=200000,  fit_intercept=True)
       