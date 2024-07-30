# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 14:16:18 2021

@author: Pedro

Auxiliary functions for:
    
    - collecting the data from excel files
    - building CNN architectures
    - computing accuracy
    - variable selection within spectrum
    - coordenate generation for spectrum selection
    - convert 2D-EEM to 1D and vice-versa, for applying CNNR_2D
    - plotting results
    - color code conversions
    
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
import joblib
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)


# def collectdata(folder):

#     monitordata = pd.read_excel(folder+'//monitordata.xlsx').set_index('Obs_ID')
#     sampledata = pd.read_excel(folder+'//sampledata.xlsx').set_index('Obs_ID')
#     fluorodata = pd.read_excel(folder+'//fluorodata.xlsx').set_index('Obs_ID').dropna().sort_index(axis=1)
#     absdata = pd.read_excel(folder+'//absdata.xlsx').set_index('Obs_ID')
#     absextdata = pd.read_excel(folder+'//absextdata.xlsx').set_index('Obs_ID')
#     no3po4data = pd.read_excel(folder+'//no3po4data.xlsx').set_index('Obs_ID')
    
#     return monitordata,sampledata,fluorodata,absdata,absextdata,no3po4data

def collectdata(folder):

    monitordata = joblib.load(folder+'/monitordata.pkl').set_index('Obs_ID')
    sampledata = joblib.load(folder+'/sampledata.pkl').set_index('Obs_ID')
    fluorodata = joblib.load(folder+'/fluorodata.pkl').set_index('Obs_ID').dropna().sort_index(axis=1)
    absdata = joblib.load(folder+'/absdata.pkl').set_index('Obs_ID')
    absextdata = joblib.load(folder+'/absextdata.pkl').set_index('Obs_ID')
    no3po4data = joblib.load(folder+'/no3po4data.pkl').set_index('Obs_ID')
    
    return monitordata,sampledata,fluorodata,absdata,absextdata,no3po4data

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
    

   
def windowgen(positions, dim, overlap):
    window_indices = []
    num_positions_in_window = []
    
    step = round(dim * (1 - overlap))
    max_x = positions[-1][0]+ int(dim/2)
    max_y = positions[-1][1]+ int(dim/2)
    
    # Slide the window over the 2D space
    for x_start in range(0, max_x - dim + 1, step):
        for y_start in range(x_start, max_y - dim + 1, step):
            x_end = x_start + dim
            y_end = y_start + dim
            
            # Find indices of positions within the current window
            indices = [i for i, (x, y) in enumerate(positions) if x_start <= x < x_end and y_start <= y < y_end]
            
            window_indices.append(indices)
            num_positions_in_window.append(len(indices))
    
    return window_indices, num_positions_in_window

def fluoro1Dgen(varxID: list, excitation_interval: list, window: int, overlap: int):
    """
    Generate 1D fluorescence spectra for specified excitation intervals with variable windows and overlaps.

    Parameters:
    varxID (list): List of column names containing the fluorescence data.
    excitation_interval (list): List of excitation wavelengths to filter.
    window (int): Size of the window (number of excitation wavelengths) for each interval.
    overlap (int): Overlap size (number of excitation wavelengths) between consecutive windows.

    Returns:
    list: List of lists, each containing the column names for the specified excitation intervals.
    """
    
    filtered_varxID_list = []
    
    start_idx = 0
    while start_idx < len(excitation_interval):
        # Determine the end index of the current window
        end_idx = min(start_idx + window, len(excitation_interval))
        
        # Get the excitation wavelengths for the current window
        current_window = excitation_interval[start_idx:end_idx]
        
        # Filter the columns for the current window
        filtered_varxID = []
        for excitation in current_window:
            excitation_str = f'EEP {excitation} / '
            filtered_varxID.extend([var for var in varxID if var.startswith(excitation_str)])
        
        if filtered_varxID:
            filtered_varxID_list.append(filtered_varxID)
        
        # Move to the next window position, taking overlap into account
        start_idx += (window - overlap)
    
    return filtered_varxID_list

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
    em = np.arange(em1,em2+step,step)
    
    iteration = 0
        
    for j in range(len(ex)):
        for i in range(len(em)):
            exc = ex[i]
            emi = em[j]   
            if ex[i] < em[j]:
                positions.append([i,j])
                waves.append([ex[i],em[j]])          
                ids.append(f'EEP {exc} / {emi} nm')
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

    # plt.pcolormesh(xnew, ynew, data1, cmap='turbo')
    # plt.colorbar()
    # plt.xlim([260,800])
    # plt.ylim([250,790])
    
    # Create a colormap using LinearSegmentedColormap
    # custom_cmap = LinearSegmentedColormap.from_list('custom_colormap', colors)
    
    # plt.pcolormesh(xnew, ynew, data1, cmap='turbo',vmin=0,vmax=1000) 
    
    # plt.pcolormesh(xnew, ynew, data1, cmap=custom_cmap, vmin = -1,vmax=1)
    
    contourf=plt.contourf(xnew, ynew, data1, #levels = [0,0.1,.2,.3,.4,.5,.6,.7,.8,.9,1],
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


def plot_partialspectrum(df2plot_ref,df2plot, spectrum_index,vmin=0,vmax=1000):
    """

    """

    # Extract the column names
    column_names = df2plot_ref.columns
    excitation_wavelengths = []
    emission_wavelengths = []
    
    # Parse the excitation and emission wavelengths from the column names
    for name in column_names:
        excitation, emission = name.split(' / ')
        excitation_wavelengths.append(int(excitation.replace('EEP ', '').replace('nm', '').strip()))
        emission_wavelengths.append(int(emission.replace('nm', '').strip()))
    
    # Get unique excitation and emission wavelengths
    unique_excitation = sorted(set(excitation_wavelengths))
    unique_emission = sorted(set(emission_wavelengths))
    
    # Create a 2D array for the original heatmap data
    data2plot = np.zeros((len(unique_excitation), len(unique_emission)))-0.001
    
    datafromdf = df2plot.loc[spectrum_index]
    
    # Fill the original heatmap data
    for i, exc in enumerate(unique_excitation):
        for j, em in enumerate(unique_emission):
            if em > exc:
                col_name = f'EEP {exc} / {em} nm'
                if col_name in df2plot.columns:
                    data2plot[i, j] = datafromdf[col_name]

    levels = [-0.1,-0.05,-0.025,-0.01,0,0.01,0.025,0.05,0.1]

    # Create the heatmap for reference data
    plt.figure(figsize=(10, 8))
    colors = [(0.0, 'cyan'),(0.25,'dodgerblue'),(0.4, 'black'),
              (0.6, 'black'), (0.75, 'red'),
              (1.0, 'yellow')]

    custom_cmap = LinearSegmentedColormap.from_list('custom_colormap', colors)
        
    contourf =plt.contourf(unique_emission,unique_excitation,
                 data2plot, cmap=custom_cmap,levels=levels,
                 origin='lower',vmax = levels[-2],vmin = levels[1])
    plt.colorbar()
    plt.contour(unique_emission,unique_excitation,data2plot, levels=contourf.levels, colors='white')#, linestyles='dashed')
    # plt.clabel(CS1, inline=1, fontsize=10, fmt='%1.1f')
    plt.xlabel('Emission Wavelength (nm)')
    plt.ylabel('Excitation Wavelength (nm)')
    plt.title(f'Original Spectrum (Spectrum Index: {spectrum_index})')
    plt.show()


def plot_partialspectrum2(df2plot_ref,df2plot, spectrum_index,vmin=0,vmax=1000):
    """

    """

    # Extract the column names
    column_names = df2plot_ref.columns
    excitation_wavelengths = []
    emission_wavelengths = []
    
    # Parse the excitation and emission wavelengths from the column names
    for name in column_names:
        excitation, emission = name.split(' / ')
        excitation_wavelengths.append(int(excitation.replace('EEP ', '').replace('nm', '').strip()))
        emission_wavelengths.append(int(emission.replace('nm', '').strip()))
    
    # Get unique excitation and emission wavelengths
    unique_excitation = sorted(set(excitation_wavelengths))
    unique_emission = sorted(set(emission_wavelengths))
    
    # Create a 2D array for the original heatmap data
    data2plot = np.zeros((len(unique_excitation), len(unique_emission)))-0.001
    
    datafromdf = df2plot.loc[spectrum_index]
    
    # Fill the original heatmap data
    for i, exc in enumerate(unique_excitation):
        for j, em in enumerate(unique_emission):
            if em > exc:
                col_name = f'EEP {exc} / {em} nm'
                if col_name in df2plot.columns:
                    data2plot[i, j] = datafromdf[col_name]

    # levels = [-0.1,-0.05,-0.025,-0.01,0,0.01,0.025,0.05,0.1]

    # Create the heatmap for reference data
    plt.figure(figsize=(10, 8))


        
    contourf =plt.contourf(unique_emission,unique_excitation,
                 data2plot, cmap='turbo',levels=5,
                 origin='lower',vmax = vmax,vmin = vmin)
    plt.colorbar()
    plt.contour(unique_emission,unique_excitation,data2plot, levels=contourf.levels, colors='white')#, linestyles='dashed')
    # plt.clabel(CS1, inline=1, fontsize=10, fmt='%1.1f')
    plt.xlabel('Emission Wavelength (nm)')
    plt.ylabel('Excitation Wavelength (nm)')
    plt.title(f'Original Spectrum (Spectrum Index: {spectrum_index})')
    plt.show()
    
def plot_partialspectrum3(ax,df2plot_ref,df2plot, spectrum_index,vmin=0,vmax=1000):
    """

    """

    # Extract the column names
    column_names = df2plot_ref.columns
    excitation_wavelengths = []
    emission_wavelengths = []
    
    # Parse the excitation and emission wavelengths from the column names
    for name in column_names:
        excitation, emission = name.split(' / ')
        excitation_wavelengths.append(int(excitation.replace('EEP ', '').replace('nm', '').strip()))
        emission_wavelengths.append(int(emission.replace('nm', '').strip()))
    
    # Get unique excitation and emission wavelengths
    unique_excitation = sorted(set(excitation_wavelengths))
    unique_emission = sorted(set(emission_wavelengths))
    
    # Create a 2D array for the original heatmap data
    data2plot = np.zeros((len(unique_excitation), len(unique_emission)))-0.001
    
    datafromdf = df2plot.loc[spectrum_index]
    # datafromdf = datafromdf[~datafromdf.index.duplicated(keep='first')]
    
    # Fill the original heatmap data
    for i, exc in enumerate(unique_excitation):
        for j, em in enumerate(unique_emission):
            if em > exc:
                col_name = f'EEP {exc} / {em} nm'
                if col_name in df2plot.columns:
                    data2plot[i, j] = datafromdf[col_name]

    # levels = [-0.1,-0.05,-0.025,-0.01,0,0.01,0.025,0.05,0.1]

    # Create the heatmap for reference data
    # ax.figure(figsize=(10, 8))


        
    contourf = ax.contourf(unique_emission,unique_excitation,
                 data2plot, cmap='turbo',levels=5,
                 origin='lower',vmax = vmax,vmin = vmin)
    
    
    ax.contour(unique_emission,unique_excitation,data2plot, levels=contourf.levels, colors='white')#, linestyles='dashed')
    
    # plt.contour(unique_emission,unique_excitation,data2plot, levels=contourf.levels, colors='white')#, linestyles='dashed')
    # plt.clabel(CS1, inline=1, fontsize=10, fmt='%1.1f')
    ax.set_xlabel('Emission Wavelength (nm)')
    ax.set_ylabel('Excitation Wavelength (nm)')
    ax.set_title(f'Original Spectrum (Spectrum Index: {spectrum_index})')

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


def hex_to_rgb(value):
    """
    Converts a HEX color code to an RGB tuple.

    Args:
        value (str): The HEX color code (e.g., '#B4FBB8').

    Returns:
        tuple: An RGB tuple (e.g., (180, 251, 184)).
    """
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


    
def mpl_color_to_rgb(code):
    """
    Converts a Matplotlib standard color code (e.g., 'C0', 'C1') to an RGB tuple.

    Args:
        code (str): The Matplotlib color code (e.g., 'C0').

    Returns:
        tuple: An RGB tuple (e.g., (0, 0, 0)).
    """
    # Get the color from Matplotlib's default color cycle
    color_list = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_index = int(code[1:]) % len(color_list)  # Handle cyclic color cycle

    # Convert the color name to RGB
    color_rgb = plt.get_cmap('tab10')(color_index)[:3]  # Get RGB values (normalized)

    # Scale the RGB values to the range [0, 255]
    # rgb_values = tuple(int(val * 255) for val in color_rgb)

    return color_rgb

    