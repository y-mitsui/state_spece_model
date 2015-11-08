# -*- coding:utf-8 -*-
import sys
import numpy
import pystan
import theano
import theano.tensor as T
import pandas as pd
import matplotlib.pyplot as plt
from hmc import HMC
import math
import time

df1 = pd.read_csv("ts.txt")
ts1 = pd.TimeSeries(df1["Sale"].as_matrix(),index=pd.to_datetime(df1["Date"]))
n=ts1.shape[0]

sample = df1["Sale"].as_matrix()
parameter = T.dvector('parameter')
y = theano.shared(sample, name='y')

#normalPdfSyntax1 = 1./(T.sqrt(2*math.pi*parameter[2])) * T.exp(-(parameter[0]-0.)**2/(2*parameter[2]))
#normalPdfSyntax2 = 1./(T.sqrt(2*math.pi*parameter[3])) * T.exp(-(parameter[1]-0.)**2/(2*parameter[3]))
#normalPdfSyntax3 = 1./(T.sqrt(2*math.pi*parameter[0])) * T.exp(-(y-parameter[4:])**2/(2*parameter[0]))
#normalPdfSyntax4 = 1./(T.sqrt(2*math.pi*parameter[1])) * T.exp(-(parameter[6:]-(2*parameter[4:-2]-parameter[5:-1]))**2/(2*parameter[1]))
#normalPdfSyntax5 = 1./(T.sqrt(2*1)) * T.exp(-(parameter[4]-0)**2/(2*1))
#normalPdfSyntax6 = 1./(T.sqrt(2*1)) * T.exp(-(parameter[5]-0)**2/(2*1))

normalPdfSyntax1 = T.log(1. / T.sqrt(2 * math.pi * parameter[2])) - 0.5 * parameter[0] ** 2 / parameter[2]
#normalPdfSyntax1 = T.log(1. / T.sqrt(2 * math.pi * 10)) - 0.5 * (parameter[0]-100) ** 2 / 10
#normalPdfSyntax2 = T.log(1. / T.sqrt(2 * math.pi * parameter[3])) - 0.5 * parameter[1] ** 2 / parameter[3]
normalPdfSyntax3 = T.log(1. / T.sqrt(2 * math.pi * parameter[0])) - 0.5 * (y-parameter[4:]) ** 2 / parameter[0]
#normalPdfSyntax3 = T.log(1. / T.sqrt(2 * math.pi * parameter[0])) - 0.5 * (y-parameter[4:]) ** 2 / parameter[0]
#normalPdfSyntax4 = T.log(1. / T.sqrt(2 * math.pi * parameter[1])) - 0.5 * (parameter[6:] - (2 * parameter[4:-2] - parameter[5:-1])) ** 2 / parameter[1]
normalPdfSyntax4 = T.log(1. / T.sqrt(2 * math.pi * parameter[1])) - 0.5 * (parameter[5:] - parameter[4:-1]) ** 2 / parameter[1]
normalPdfSyntax5 = T.log(1. / T.sqrt(2 * math.pi * 1)) - 0.5 * (parameter[4] - 4000) ** 2 / 1
#normalPdfSyntax6 = T.log(1. / T.sqrt(2 * math.pi * 1)) - 0.5 * (parameter[5] - 4000) ** 2 / 1
normalPdfSyntax7 = T.log(1. / T.sqrt(2 * math.pi * 0.1)) - 0.5 * (parameter[2] - 0) ** 2 / 0.1
normalPdfSyntax8 = T.log(1. / T.sqrt(2 * math.pi * 10)) - 0.5 * (parameter[3] - 0) ** 2 / 10

normalPdfSyntaxSP = T.log(1. / T.sqrt(2 * math.pi * 10.)) - 0.5 * (y-parameter[4:]) ** 2 / 10.

#posteriorSyntax = T.log(normalPdfSyntax1) + T.log(normalPdfSyntax2) + T.sum(T.log(normalPdfSyntax3)) + T.sum(T.log(normalPdfSyntax4)) + T.log(normalPdfSyntax5) + T.log(normalPdfSyntax6)
#posteriorSyntax = normalPdfSyntax1 + normalPdfSyntax2 + T.sum(normalPdfSyntax3) + T.sum(normalPdfSyntax4) + normalPdfSyntax5 + normalPdfSyntax6 + normalPdfSyntax7 + normalPdfSyntax8
#posteriorSyntax =  normalPdfSyntax1 + normalPdfSyntax2 + normalPdfSyntax7 + normalPdfSyntax8 + T.sum(normalPdfSyntax3)
posteriorSyntax = T.sum(normalPdfSyntax3) + T.sum(normalPdfSyntax4) #+ normalPdfSyntax1 + normalPdfSyntax2 + T.log(normalPdfSyntax5)



posterior = theano.function(inputs=[parameter], outputs=posteriorSyntax)
param = [1000.] * 3 + [numpy.average(sample)] * sample.shape[0]
print posterior(numpy.array([0.] + param))
#sys.exit(1)s
gPosteriorSyntax = T.grad(cost=posteriorSyntax, wrt=parameter)
gPosterior = theano.function(inputs=[parameter], outputs=gPosteriorSyntax)

def callPosterior(param,argument):
#    print param
    r = posterior(param)
#    print r
    return r
def callGPosterior(param,argument):
    r =  gPosterior(param)
#    print param
#    print r
#    print "------"
#    time.sleep(1)
    return r
    

range_parameter = [[1e-10,1000000]] * 4 + [[-10,12000]] * sample.shape[0]
init_theta =  [1.] * 4 + [numpy.average(sample)] * sample.shape[0]
init_theta = numpy.array([min(max(x,minI),maxI) for x,(minI,maxI) in zip(init_theta,range_parameter)])
context = HMC(callPosterior, callGPosterior, None, init_theta, range_parameter)
estimated_parameter = context.sampling(iter=4000,burn_in=3000,iter_leapfrog=40)
print numpy.average(estimated_parameter,axis=0)
pd.DataFrame({"estimate":numpy.average(estimated_parameter,axis=0)[4:],"original":sample}).plot()
plt.show()
