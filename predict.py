import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

#Feed the data from csv file
data_feed=pd.read_csv('../predict_files/data_stocks.csv',usecols = [0,1],skiprows = [0],header=None)
#print(data_feed) #Print the values(Debugging only)

#Plotting
x=data_feed[0] #Date Value
y=data_feed[1] #Price Value

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
