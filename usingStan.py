# -*- coding:utf-8 -*-
import sys
import numpy
import pystan
import random
import pandas as pd
import matplotlib.pyplot as plt

df1 = pd.read_csv("ts.txt")
ts1 = pd.TimeSeries(df1["Sale"].as_matrix(),index=pd.to_datetime(df1["Date"]))
n=ts1.shape[0]
training_data = dict(N=n,y=df1["Sale"].as_matrix())
fit = pystan.stan(file='trend.stan', data=training_data, iter=10000, chains=1)
print "------- hierarchical bayse estimate ------------"
print fit
ar_params = fit.get_posterior_mean()
#df.plot()
print ar_params[:n].shape
print df1["Sale"].as_matrix().shape
pd.DataFrame({"a":df1["Sale"].as_matrix(),"b":ar_params[:n,0]},index=pd.to_datetime(df1["Date"])).plot()
#plt.plot(ar_params[:n])
plt.show()
