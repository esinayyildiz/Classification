import numpy as np
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


def MINST():
 MNIST = load_digits()

 train_data_0,test_data_0,train_data_1,test_data_1 = train_test_split(np.array(MNIST.data),MNIST.target,test_size=0.33)
 train_data_0,val_Data,train_data_1,val_Label = train_test_split(train_data_0,train_data_1,test_size=0.15)




 print("training data sets: {}".format(len(train_data_1)))
 print("validation data sets: {}".format(len(val_Label)))
 print("testing data sets: {}".format(len(test_data_1)))
    

 for k in np.arange(1,40,3):
    
    kNN = KNeighborsClassifier(n_neighbors=k)
    kNN.fit(train_data_0,train_data_1)
    
    # evaluate the model and update the accuracies list
    RESULT = kNN.score(val_Data, val_Label)
    print("k=%d, accuracy=%.3f%%" % (k, RESULT * 100))


MINST()