import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#Feed the data from csv file
data_feed=pd.read_csv('../predict_files/data_stocks.csv')#,usecols = [0,1],skiprows = [0],header=None)
#print(data_feed) #Print the values(Debugging only)

# Drop date variable
data_feed = data_feed.drop(['DATE'], 1)

# Dimensions of dataset
n = data_feed.shape[0]
p = data_feed.shape[1]


# Make data a np.array
data_feed = data_feed.values
#print(data_feed)

train_start = 0
train_end = int(np.floor(0.8*n))
test_start = train_end + 1
test_end = n
data_train = data_feed[np.arange(train_start, train_end), :]
data_test = data_feed[np.arange(test_start, test_end), :]

X_train = data_train[:, 1:]
y_train = data_train[:, 0]
X_test = data_test[:, 1:]
y_test = data_test[:, 0]
print(y_test)
