# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 16:26:05 2023

@author: Pedro
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt
from matplotlib import rcParams

def getEEMpositions3(ex1,ex2,em1,em2,varxID):
    
    positions = [] # EEP coordinates in the EEM
    waves = []    
    ex = np.arange(ex1,ex2+5,5)
    em = np.arange(em1,em2+5,5)
    
    iteration = 0
        
    for j in range(len(ex)):
        for i in range(len(em)):
            if ex[i] < em[j]:
                positions.append([i,j])
                waves.append([ex[i],em[j]])                
                iteration +=1
    
    translations = pd.DataFrame(columns = ['ex','em'],data=waves)
    translations['id'] = varxID
    translations['coords'] = positions
          
    return positions,translations

def datatransform_testset(X_test,transformation,mean,standard_dev,log_shift):
    
    X_processed = X_test.copy()
    
    if transformation == 'Ori':       
        print("No transformation applied")
        
    elif transformation == 'MCSN':
        X_processed = (X_processed - mean)*1/standard_dev
                
    elif transformation == 'LogBoth':  
        X_processed[X_processed<0.001] = 0.001
        X_processed = np.log(X_processed)
        
        
    return X_processed

def plotasEEM(newpositions,EEM4plot):
    
    mapacoefs = np.zeros((109,109))
    mapacoefs[:] = 0
    
    ex = np.arange(250,795,5)
    em = np.arange(260,805,5)
    
    xplot,yplot = np.meshgrid(em,ex)
    
    for iteration in range(len(newpositions)):
        
        mapacoefs[newpositions[iteration][0],newpositions[iteration][1]] = EEM4plot[iteration]

    xplot = em
    yplot = ex
    z = mapacoefs
    
    f = interp2d(xplot, yplot, z, kind='linear')
    ynew = np.arange(250,795,5)
    xnew = np.arange(260,805,5)
    data1 = f(xnew,ynew)
    
    contourf=plt.contourf(xnew, ynew, data1,levels = 10,
                            cmap= 'turbo',vmax =1000,vmin = 0)
    plt.colorbar()
    plt.contour(xnew, ynew, data1, levels=contourf.levels, colors='white')
    
    
import seaborn as sns

def r2(y_true,y_pred):
    RSS = np.sum((y_true - y_pred.reshape(y_true.shape))**2)
    TSS = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - RSS/TSS
    # print(r2)
    return r2

def rmse(y_true,y_pred):
    RSS = np.sum((y_true - y_pred.reshape(y_true.shape))**2)
    rmse = np.sqrt(RSS/len(y_true))
    return rmse

def plotAccScatters(y,output2predict):
    
    
    y_test, y_pred =  y[output2predict],y[output2predict+" - Model"]
    
    yplotmax = np.max(y_test)
    
     
    # y = y[y['Assay_ori'].isin(['F2'])]
    
    # Define markers and colors
    markers = ['o','s','^','d']
    colors = ['C0','C1','C2','C4']
    colors = [(1.0, 0.8352941176470589, 0.4745098039215686),
              (0.7490196078431373, 0.5647058823529412, 0),
              (0.4980392156862745, 0.3764705882352941, 0),
              (0.3980392156862745, 0.2764705882352941, 0)]
    
    
    assay = "Assay_ori"
    
    unique_mlalg = y[assay].sort_values().unique()
    

    # unique_mlalg = [str(year) for year in unique_mlalg]
    
    # Create a dictionary for markers and colors
    marker_map = {alg: marker for alg, marker in zip(unique_mlalg, markers)}
    color_map = {alg: color for alg, color in zip(unique_mlalg, colors)}
    
    for alg in unique_mlalg:
        marker = marker_map[alg]
        color = color_map[alg]
        subset = y[y[assay] == alg]
    
        sns.scatterplot(data=subset, x=output2predict, y=output2predict + ' - Model',
                        marker=marker, color=color,label=str(alg),legend=False)
    
    stdy = np.std(y_test)
    # yplotmax = np.max(y_test)
    
    plt.plot([0,yplotmax],[0,yplotmax],'--',color='k',linewidth=0.75) # Y = PredY line
    plt.plot([0,yplotmax],[stdy,yplotmax+stdy],'--',color='k',linewidth=0.75)
    plt.plot([0,yplotmax],[-stdy,yplotmax-stdy],'--',color='k',linewidth=0.75)
    
    plt.xlim([0,yplotmax])
    plt.ylim([0,yplotmax]) 
    plt.ylabel(output2predict+' - Model')
    plt.xlabel(output2predict+' - Experimental')
    
    plt.show()
    
