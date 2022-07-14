#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
import sys

def Logistic(x, K, N_o, r):
    y = K / (1 + ((K-N_o)/N_o)*np.exp(-r*x))
    return y

def fit_and_report(func, x, y, guess=False):
    if guess:
        parameters, covariance = curve_fit(Logistic, x, y, p0=guess)
    else:
        parameters, covariance = curve_fit(Logistic, x, y, p0=guess)
        
    print(f'K={parameters[0]:.3e} +/- {covariance[0,0]*np.sqrt(len(x)):.3e}')
    print(f'N_o={parameters[1]:.3e} +/- {covariance[1,1]*np.sqrt(len(x)):.3e}')
    print(f'r={parameters[2]:.3f} +/- {covariance[2,2]*np.sqrt(len(x)):.3f}')
    print(f'D.T.={np.log(2)/parameters[2]}')
    return parameters, covariance

data = pd.read_csv(sys.argv[1], index_col='time', header=0)

print(data.columns) 
concatetimes = np.concatenate([data.index.values,data.index.values,data.index.values])
concatvalues = np.concatenate([data.A.values,data.B.values,data.C.values])

guess = [concatvalues[-1],concatvalues[0],2/concatetimes[-1]]
x = np.linspace(np.min(concatetimes),np.max(concatetimes),13)
plt.scatter(concatetimes,concatvalues)
parms, covar = fit_and_report(Logistic,concatetimes, concatvalues, guess)
plt.errorbar(x, Logistic(x,*parms), yerr=data.std(axis=1))

plt.ylabel('Absorbance')
plt.xlabel('Time[hours]')
plt.savefig(sys.argv[1].strip('.csv')+'.jpg') 
