import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

#Feed the data from csv file
data_feed=pd.read_csv('../predict_files/data_stocks.csv',usecols = [0,1],skiprows = [0],header=None)
#print(data_feed) #Print the values(Debugging only)

#data Parameterization
data_feed = data_feed.values
#plt.plot(x,y) #Plot for Debugging
#plt.show()  #Plot for Debugging

n = data_feed.shape[0]
p = data_feed.shape[1]

#Training and Testing Parameters
train_start = 0
train_end = int(np.floor(0.8*n))
test_start = train_end + 1
test_end = n
data_train = data_feed[np.arange(train_start, train_end), :]
data_test = data_feed[np.arange(test_start, test_end), :]

# Scale data
scaler = MinMaxScaler()
scaler.fit(data_train)
data_train = scaler.transform(data_train)
data_test = scaler.transform(data_test)

# Build X and y
X_train = data_train[:, 1:]
y_train = data_train[:, 0]
X_test = data_test[:, 1:]
y_test = data_test[:, 0]

#plt.plot(y_train,X_train) #Plot for Debugging
#plt.show()  #Plot for Debugging
