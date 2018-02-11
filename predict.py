import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

#Feed the data from csv file
data_feed=pd.read_csv('data_stocks.csv',usecols = [0,1],skiprows = [0],header=None)
#print(data_feed) #Print the values(Debugging only)

#Plotting
x=data_feed[0] #Date Value
y=data_feed[1] #Price Value

#plt.plot(x,y) #Plot for Debugging
#plt.show()  #Plot for Debugging
