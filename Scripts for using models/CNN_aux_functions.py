# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 14:16:18 2021

@author: Pedro

Necessary functions for running CNN Deep Learning

"""

import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RepeatedKFold
import matplotlib.pyplot as plt

import tensorflow as tf
from keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Lambda
from tensorflow.keras.layers import Conv1D, Conv2D, Reshape, MaxPooling2D
from tensorflow.keras.callbacks import ReduceLROnPlateau
from keras.layers.noise import GaussianNoise
import tensorflow.keras.optimizers as heyhey
from scipy.interpolate import interp2d

from matplotlib.colors import LinearSegmentedColormap
physical_devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)


def collectdata():

    monitordata = pd.read_excel('monitordata_Ax_v3.xlsx').set_index('Obs_ID')
    sampledata = pd.read_excel('sampledata_Ax.xlsx').set_index('Obs_ID')
    fluorodata = pd.read_excel('fluorodata_Ax_v3.xlsx').set_index('Obs_ID')
    # fluorodata = pd.read_excel('fluorodata_Ax_bigcuv.xlsx').set_index('Obs_ID')
    absdata = pd.read_excel('absdata_Ax.xlsx').set_index('Obs_ID')
    absextdata = pd.read_excel('absextdata_Ax_v4.xlsx').set_index('Obs_ID')
    
    return monitordata,sampledata,fluorodata,absdata,absextdata

def collectdata_micbio():

    monitordata = pd.read_excel('monitordata_Nassay.xlsx').set_index('Obs_ID')
    sampledata = pd.read_excel('sampledata_Nassay.xlsx').set_index('Obs_ID')
    fluorodata = pd.read_excel('fluorodata_Nassay.xlsx').set_index('Obs_ID')
    absdata = pd.read_excel('absdata_Nassay.xlsx').set_index('Obs_ID')
    absextdata = pd.read_excel('absextdata_Nassay.xlsx').set_index('Obs_ID')
    
    return monitordata,sampledata,fluorodata,absdata,absextdata

#The model
def make_model(input_dim,hyperparameters,MLproblem):
    model = Sequential()
    
    C1_K = hyperparameters.get('C1_K')
    C2_K = hyperparameters.get('C2_K')
    C1_S = hyperparameters.get('C1_S')
    C2_S = hyperparameters.get('C2_S')
    activation = hyperparameters.get('activation')
    DROPOUT = hyperparameters.get('DROPOUT')
    DENSE = hyperparameters.get('DENSE')
    
    ## Adding a bit of GaussianNoise also works as regularization
    # model.add(GaussianNoise(0.05, input_shape=(input_dim,)))
    
    # Use a Lambda layer to apply Gaussian noise during inference only
    model.add(Lambda(lambda x: x + K.random_normal(shape=K.shape(x), mean=0.05, stddev=0.05),
                      input_shape=(input_dim,)))    
    
    #First two is number offRe filter + kernel size
    model.add(Reshape((input_dim, 1) ))
    model.add(Conv1D(C1_K, (C1_S), activation=activation, padding="same"))
    model.add(Conv1D(C2_K, (C2_S), padding="same", activation=activation))
    model.add(Flatten())
    model.add(Dropout(DROPOUT))
    model.add(Dense(DENSE, activation=activation))
    
    if MLproblem == 'Classification':
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam',
                      metrics=[
                          tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                          tf.keras.metrics.Precision(name='precision'),
                          tf.keras.metrics.Recall(name='recall')
                      ])
    else:
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mse', optimizer=heyhey.Adadelta(learning_rate=0.05))

    return model


def make_model_2D(input_shape, hyperparameters,MLproblem):
    model = Sequential()

    C1_K = hyperparameters.get('C1_K')
    C2_K = hyperparameters.get('C2_K')
    C1_S = hyperparameters.get('C1_S')
    C2_S = hyperparameters.get('C2_S')
    activation = hyperparameters.get('activation')
    DROPOUT = hyperparameters.get('DROPOUT')
    DENSE = hyperparameters.get('DENSE')

    # Use a Lambda layer to apply Gaussian noise during inference only
    model.add(Lambda(lambda x: x + K.random_normal(shape=K.shape(x), mean=0.05, stddev=0.05),
                      input_shape=input_shape))    
    
    model.add(Conv2D(C1_K, (C1_S, C1_S), activation=activation, padding="same"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(C2_K, (C2_S, C2_S), padding="same", activation=activation))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dropout(DROPOUT))
    model.add(Dense(DENSE, activation=activation))
    
    if MLproblem == 'Classification':
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam',     
                      metrics=[
                                tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                                tf.keras.metrics.Precision(name='precision'),
                                tf.keras.metrics.Recall(name='recall')
            ])
    else:
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mse', optimizer=heyhey.Adadelta(learning_rate=0.05))


    return model



#Some metrics

def r2(y_true,y_pred):
    RSS = np.sum((y_true - y_pred.reshape(y_true.shape))**2)
    TSS = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - RSS/TSS
    r2=float(r2)
    # print(r2)
    return r2

def rmse(y_true,y_pred):
    RSS = np.sum((y_true - y_pred.reshape(y_true.shape))**2)
    rmse = np.sqrt(RSS/len(y_true))
    rmse = float(rmse)
    return rmse

def PEVcalc(y_true,y_pred):
    AR = np.abs(y_true - y_pred.reshape(y_true.shape))
    PEV = np.mean(np.divide(AR,y_true))*100
    # print('PEV(%) =',round(PEV,1))
    return PEV

def linearReg(y,x):
    
    model = LinearRegression()
    x = x.reshape(-1,1)
    y = y.reshape(-1,1)
    model.fit(x,y)
    r_sq = model.score(x,y)
    # intercept = model.intercept_
    # coef = model.coef_
    # print('coefficient of determinations: ',r_sq)
    # print('intercept: ',intercept)
    # print('slope: ',coef)
    return r_sq
    
 #K-fold cross-validation

def evaluate_model(X,y,input_dim,C1_K,C2_K,C1_S,C2_S,activation,DROPOUT,DENSE):
    results = list()

 	# define evaluation procedure
    cv = RepeatedKFold(n_splits=6, n_repeats=1, random_state=1)
    
    # define learnig rate dynamics
    rdlr=ReduceLROnPlateau(patience=5,factor=0.5,min_lr=1e-6,
    monitor='val_loss',verbose=1)
    
 	# enumerate folds
    for train_ix, test_ix in cv.split(X):
        
  		# prepare and augment data
        X_train, X_test = X[train_ix], X[test_ix]
        y_train, y_test = y[train_ix], y[test_ix]
        
        y_train_aug = np.repeat(y_train, repeats=30, axis=0) #y_train is simply repeated
        
        shift = np.std(X)*0.1
        
        X_train_aug = np.repeat(X_train, repeats=30, axis=0)
        X_train_aug = dataaugment(X_train_aug, betashift = shift, slopeshift = 0.05, multishift = shift)

                
    		# define model
        model = make_model(input_dim,C1_K,C2_K,C1_S,C2_S,activation,
                            DROPOUT,DENSE)
  		# fit model
        model.fit(X_train_aug, y_train_aug, epochs=30, batch_size=8,
                      validation_data=(X_test, y_test), callbacks=[rdlr])
        
        # evaluate model on test set
        mae = model.evaluate(X_test, y_test, verbose=0)
        pev = PEVcalc(y_test,model.predict(X_test))
        pearsonR2 = linearReg(y_test, model.predict(X_test))
        regularR2 = r2(y_test,model.predict(X_test))
        
        # store result
        print('mae= ',mae)
        print('pev = ',pev)
        print('r2 = ',regularR2)
        print('pearson = ',pearsonR2)
        
        # results.append([mae,pev,regularR2,pearsonR2])
        
        stdy = np.std(y)
                     
        plt.plot([0,np.max(y_train)],[0,np.max(y)],'c--') # Y = PredY line
        plt.plot([0,np.max(y_train)],[stdy,np.max(y)+stdy],'c--')
        plt.plot([0,np.max(y_train)],[-stdy,np.max(y)-stdy],'c--')
        
        plt.scatter(y_train, model.predict(X_train))
        plt.scatter(y_test, model.predict(X_test))
        
        plt.ylabel('Model')
        plt.xlabel('Experimental')
        # plt.title(output.replace('/','_')+' transf '+str(transformation))
        
        plt.show()
    
    results = pd.DataFrame(results, columns = ['mae','pev','r2','pearson'])
    
    return results
   
def windowgen(positions,dim,overlap):
    
    windownum = 0
    dim -= 1
    numberofvariables = []
    index_all = []
    index = []

    for j in range(0,109-dim,round(dim*(1-overlap))):
        
        for i in range(j,109-dim,round(dim*(1-overlap))):
          
            llimit1 = j
            llimit2 = i;
            ulimit1 = llimit1 + dim
            ulimit2 = llimit2 + dim
            
            index = []
            
            for iter1 in range(len(positions)):
        
              if llimit1<=positions[iter1][0] and positions[iter1][0]<=ulimit1 and \
                 llimit2<=positions[iter1][1] and positions[iter1][1]<=ulimit2:
                
                index.append(iter1)
        
            windownum += 1
            index_all.append(index)
            numberofvariables.append(len(index))
          
    return index_all,numberofvariables

def windowgen_linear(spectrum_length,dim,overlap):
    
    index_all = []
    windownum = -1
    numberofvariables = []
      
    for i in range(0,(spectrum_length-dim),round(dim*(1-overlap))):
        
        llimit1 = i
        ulimit1 = llimit1 + dim
        
        index = list(range(llimit1,ulimit1))
    
        windownum += 1
        index_all.append(index)
        len(index)
        numberofvariables.append(len(index))
    
          
    return index_all,numberofvariables

def getEEMpositions(x,ex1,ex2,em1,em2):
    
    positions = [] # EEP coordinates in the EEM

    
    ex = np.arange(ex1,ex2+5,5)                           
    em = np.arange(em1,em2+5,5)
    
    iteration = 0
        
    for j in range(len(ex)):
        for i in range(len(em)):
            if ex[i] < em[j]:
                positions.append([i,j]);

                iteration +=1
                
    return positions

def getEEMpositions2(ex1,ex2,em1,em2,step):
    
    positions = [] # EEP coordinates in the EEM
    waves = [] 
    ids = []
    ex = np.arange(ex1,ex2+step,step)
    em = np.arange(em1,em2+5,5)
    
    iteration = 0
        
    for j in range(len(em)):
        for i in range(len(ex)):
            if ex[i] < em[j]:
                positions.append([i,j])
                waves.append([ex[i],em[j]])   
                ids.append('EEP '+str(ex[i])+' / '+str(em[j])+' nm')
                iteration +=1
                
    translations = pd.DataFrame(columns = ['ex','em'],data=waves)
    translations['coords'] = positions      
    translations['ids'] = ids    
    
    return positions,translations

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

def inputselectf(selectiontres,inclusion):
    
    index = []
    
    for iteration in range(len(selectiontres)):
        
        if selectiontres[iteration] >= inclusion:
            index.append(iteration)
            
    return index



def autoinputselect2(selectiontres,VIPstep,Xori,xpositions,minimal_numberofvariables):
    
    ini_inclusion = 0.95 # min(selectiontres)
    end_inclusion = max(selectiontres)
    step = (end_inclusion-ini_inclusion)*VIPstep/1000

    numberofvariables_ori = len(selectiontres)

    # Plot variable selection
    
    numberofvariables = numberofvariables_ori
    storenumberofvariables = []
    
    def f(t):
        out = len(inputselectf(selectiontres, t))
        return out
    
    inclusion_plot = []
    for inclusion in np.arange(ini_inclusion,end_inclusion,step):
      
      if f(inclusion) <= numberofvariables*0.975:
        storenumberofvariables.append(f(inclusion))
        inclusion_plot.append(inclusion)
      
      if f(inclusion) == 0:
        f(inclusion)
        break
    
    plt.scatter(inclusion_plot,storenumberofvariables,s=12)
    plt.title('Variable selection dynamics')
    plt.xlabel('threshold')
    plt.ylabel('spectral variables selected')
    plt.show()

    # Generation of groups of EEPs

    iteration = 0
    index_all = []
    
    numberofvariables = numberofvariables_ori
    
    index_all.append(list(range(numberofvariables)))
    
    for inclusion in np.arange(ini_inclusion,end_inclusion,step):
  
      #print(inclusion)
        
      if len(inputselectf(selectiontres,inclusion)) <= numberofvariables*0.975:
        
        iteration += 1
        numberofvariables = len(inputselectf(selectiontres,inclusion))
        # print('Number of variables = ',numberofvariables)
        if numberofvariables <= minimal_numberofvariables:
          print("Done")
          break

        index_all.append(inputselectf(selectiontres,inclusion))
        storenumberofvariables.append(numberofvariables)
    
    return index_all,storenumberofvariables
  
def windowgen89(positions,dim,overlap):
    
    windownum = 0
    dim -= 1
    numberofvariables = []
    index_all = []
    index = []

    for j in range(0,89-dim,round(dim*(1-overlap))):
        
        for i in range(j,89-dim,round(dim*(1-overlap))):
          
            llimit1 = j
            llimit2 = i;
            ulimit1 = llimit1 + dim
            ulimit2 = llimit2 + dim
            
            index = []
            
            for iter1 in range(len(positions)):
        
              if llimit1<=positions[iter1][0] and positions[iter1][0]<=ulimit1 and \
                 llimit2<=positions[iter1][1] and positions[iter1][1]<=ulimit2:
                
                index.append(iter1)
        
            windownum += 1
            index_all.append(index)
            numberofvariables.append(len(index))
          
    return index_all,numberofvariables

def plotasEEM(newpositions,EEM4plot):
    
    mapacoefs = np.zeros((109,109))
    mapacoefs[:] = 0
    print('hello')
    ex = np.arange(250,795,5)
    em = np.arange(260,805,5)
    
    xplot,yplot = np.meshgrid(em,ex)
    
    for iteration in range(len(newpositions)):
        
        mapacoefs[newpositions[iteration][0],newpositions[iteration][1]] = EEM4plot[iteration]      

    xplot = em
    yplot = ex
    z = mapacoefs
    
    f = interp2d(xplot, yplot, z, kind='linear')
    ynew = np.arange(250,795,0.5)
    xnew = np.arange(260,805,0.5)
    data1 = f(xnew,ynew)

    
    contourf=plt.contourf(xnew, ynew, data1, levels = 7,
                            cmap= 'turbo')
    
    
    plt.colorbar()
    plt.contour(xnew, ynew, data1, levels=contourf.levels, colors='white')
 
def reconstructEEM(newpositions,EEM1d):
    
    mapacoefs = np.zeros((109,109))
    mapacoefs[:] = 1e-6
    
    for iteration in range(len(newpositions)):
        
        mapacoefs[newpositions[iteration][0],newpositions[iteration][1]] = EEM1d[iteration]        

    return mapacoefs

def reconstructEEMdatabase(X,newpositions):
    
    n = len(X)
    X_2D = np.empty((n, 109, 109))
   
    for row in range(n):
        X_2D[row] = reconstructEEM(newpositions,X[row,:])
    
     
    X_2D = np.expand_dims(X_2D, axis=-1)
   
    return X_2D
    
def plotasEEM2(ax,newpositions,EEM4plot,cmap):
    
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
    xnew = np.arange(260,805,1)
    ynew = np.arange(250,795,1)
    data1 = f(xnew,ynew)
    contourf=ax.contourf(xnew, ynew, data1,levels = 7,
                            cmap= 'turbo')
    # ax.colorbar()
    ax.contour(xnew, ynew, data1, levels=contourf.levels, colors='white')

    
def plotasEEM3(ax,newpositions,EEM4plot):
    
    mapacoefs = np.zeros((89,89))
    mapacoefs[:] = -50
    
    ex = np.arange(250,695,5)
    em = np.arange(260,705,5)
    
    xplot,yplot = np.meshgrid(em,ex)
    
    for iteration in range(len(newpositions)):
        
        mapacoefs[newpositions[iteration][0],newpositions[iteration][1]] = EEM4plot[iteration]      

    xplot = em
    yplot = ex
    z = mapacoefs
    
    f = interp2d(xplot, yplot, z, kind='linear')
    xnew = np.arange(260,705,0.5)
    ynew = np.arange(250,695,0.5)
    data1 = f(xnew,ynew)

    ax.pcolormesh(xnew, ynew, data1, cmap='turbo',vmin=-50,vmax=1000)


def plotresultsbyday(aux,output,days):
    
    # aux is the sample data plus the output data, output is the name of the output
    
    data_mean = []
    data_max = []
    data_min = []
    data_std = []
    
    for day in days:
        data_mean.append(aux[aux['Day']==day][output].mean())
        data_max.append(aux[aux['Day']==day][output].max())
        data_min.append(aux[aux['Day']==day][output].min())
        data_std.append(aux[aux['Day']==day][output].std())
        
    # df = pd.DataFrame({'Batch_ID' : aux['Batch_ID'],
    #                     'DAI' : aux['Day'],
    #                    'Output': aux[output]})
    
    # model = ols('Output ~ DAI', data=df).fit()

    # aov_table = sm.stats.anova_lm(model, typ=2)

    return data_mean,data_std,data_max,data_min#,aov_table

def maxk(inputlist,k):
    
    outputlist = []
    outputlist_ind = []
    inputlist_work = inputlist.copy()
    
    while len(outputlist) < k:
    
        max_k = max(inputlist_work)
        max_k_ind = inputlist.index(max_k)
        outputlist.append(max_k)
        outputlist_ind.append(max_k_ind)
        
        inputlist_work.remove(max_k)
     
    if k == 1:
        outputlist = outputlist[0]
        outputlist_ind = outputlist_ind[0]
        
    return outputlist,outputlist_ind


def plotasEEMcontour(positions,EEM4plot,levels):
       
    # ex = np.arange(540,735,5)
    # em = np.arange(550,745,5)
    
    ex = np.arange(250,795,5)
    em = np.arange(260,805,5)
    
    mapacoefs = np.zeros((len(ex),len(em)))
    mapacoefs[:] = 0
    
    xplot,yplot = np.meshgrid(em,ex)
    
    for iteration in range(len(positions)):
        
        mapacoefs[positions[iteration][0],positions[iteration][1]] = EEM4plot[iteration]      

    xplot = em
    yplot = ex
    z = mapacoefs
    
    f = interp2d(xplot, yplot, z, kind='linear')
    ynew = np.arange(250,791,1)
    xnew = np.arange(260,801,1)
    data1 = f(xnew,ynew)
    
    colors = [(0.0, 'cyan'),(0.25,'dodgerblue'),(0.4, 'black'),
              (0.6, 'black'), (0.75, 'red'),
              (1.0, 'yellow')]

    custom_cmap = LinearSegmentedColormap.from_list('custom_colormap', colors)
    
    
    contourf = plt.contourf(xnew, ynew, data1, levels = levels, #[-0.023,-0.017,-0.011,-0.005,0,0.005,0.011,0.017,0.023],
                            cmap=custom_cmap,vmax = levels[-2],vmin = levels[1])
    plt.colorbar()
    plt.contour(xnew, ynew, data1, levels=contourf.levels, colors='black')
                

    # plt.pcolormesh(xnew, ynew, data1, cmap='turbo',vmin=0,vmax=1000)
    # contourf = plt.contourf(xnew, ynew, data1,levels = 10,
    #                         cmap='turbo')
    # plt.colorbar()
    # plt.contour(xnew, ynew, data1, levels=contourf.levels, colors='black')
    
    plt.xlim([260,800])
    plt.ylim([250,790])


    
    
    