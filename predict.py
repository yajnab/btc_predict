import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

#Feed the data from csv file
data_feed=pd.read_csv('data_stocks.csv',usecols = [0,1],skiprows = [0],header=None)
#print(data_feed) #Print the values(Debugging only)

