import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def kalman(y,x0,A,b,c,sigma_v,sigma_w):
    """
    x(k+1) = Ax(k) + bv(k)
    y(k)   = c'x(k) + w(k)
    """
    n_dimention = x0.shape[0]
    x = x0
    result_x = []
    Sigma = np.ones((n_dimention, n_dimention)) * 200000
    for each_y in y:
        #prediction
        x = A * x 
        Sigma = A * Sigma * A.T + sigma_v * (b * b.T)
        print Sigma
        #filter
        gain = Sigma * c / ( c.T * Sigma * c  + sigma_w)
        x = x + gain * ( each_y - c.T * x)
        Sigma = (np.eye(n_dimention) - gain * c.T) * Sigma
        result_x.append(x)
    return np.array(result_x)

df1 = pd.read_csv("ts.txt")
y = df1["Sale"].as_matrix()

A = np.matrix(np.eye(1))
b = np.matrix([[1]])
c = np.matrix([[1.]])
r = kalman(y,np.matrix([[np.average(y)]]),A,b,c,1,np.std(y)**2)
r2 = np.array([x[0,0] for x in r])
pd.DataFrame({"estimate":r2,"original":y}).plot()
plt.show()


A = np.matrix([[2,-1],[1,0]])
b = np.matrix([[1.],[0]])
c = np.matrix([[1.],[0]])
r = kalman(y,np.matrix([[0.],[0.]]),A,b,c,1,np.std(y)**2)
r2 = np.array([x[0,0] for x in r])
pd.DataFrame({"estimate":r2,"original":y}).plot()
plt.show()

