# -*- coding: utf-8 -*-
"""
Created on Tue May  2 11:29:37 2023

@author: Pedro

Script for running models developed in Brandão et al. 2024 

Any model of tables 3.1 and 3.2 can be used!

validation_data should be organized as the files "Fxppm_validation.xlsx"
and "CCMmL_validation.xlsx" are, i.e., with labels "Assay_ori" and "Day",
index "Obs_ID"

"""

# Import required libraries

import joblib,os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from aux_functions import plotasEEM,datatransform_testset
import math
import numpy as np
from CNN_aux_functions import reconstructEEMdatabase
from tensorflow.keras.models import load_model

# This is for character conversion for naming files without special characters
SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
Trainsub = str.maketrans("t", "ₜ")
badchars = ["(", ")", "/", " ", "."]

#%%

# ============================ USER INPUTS ================================ #

modeldata_directory = "Model data"

# validation_data= "CCMmL_validation.xlsx" 
validation_data= "Fxppm_validation.xlsx" 


spectradatafile = "fluorodata_4test.xlsx" 
# spectradatafile = "absdata_4test.xlsx" 


output2predict = "Fx (ppm)" # "CC (M/mL)"
model = "Fx/F-2"

sample2plot = 1

plot_results_withspectra = False

name4results = "test.xlsx"
folder4figures = "results"

# ========================================================================= #


# User entries are renamed to find the model files

output_name = output2predict

for i in badchars:
    output_name = output_name.replace(i, "")

varsel_directory = modeldata_directory + "//"+ output_name+'_Optimal_VarSel'



#%%

# Load the data to predict

X_ori = pd.read_excel(spectradatafile).set_index("Obs_ID").dropna()
X = X_ori.copy()

model_dictionary = pd.read_excel("model_ID_dictionary.xlsx").set_index("Model_ID")

obsID = list(X.index)


# Load the data on the model architecture

maindirectory = os.getcwd()
modelname = model_dictionary.loc[model,:].iloc[0].split("_")


# Unpack properties of model from the dictionary translation

if len(modelname) == 6:
    output_name,MLalg,spectrum,transform,varsel,varsel2 = modelname
    varsel = varsel+"_"+varsel2
elif len(modelname) == 7:
    output_name,MLalg,MLalg2,spectrum,transform,varsel,varsel2 = modelname
    MLalg += "_"+MLalg2
    varsel = varsel+"_"+varsel2
else:
    output_name,MLalg,spectrum,transform,varsel = modelname    
    
    

modeldata_directory += "//"+output_name+'_'+MLalg
os.chdir(modeldata_directory)

modelname = model_dictionary.loc[model,:].iloc[0]
print(modelname)

if MLalg == 'CNNR' or MLalg == 'CNNR_2D':
    model_optimal = load_model(modelname+'_model_optimal')
else:
    model_optimal = joblib.load(modelname+'_model_optimal.pkl')
    
transformation = joblib.load(modelname+'_transformation.pkl')
hyperparameters_tuned = joblib.load(modelname+'_hyperparams_tuned.pkl')
X_mean = joblib.load(modelname+'_Xmean.pkl')
X_standard_dev = joblib.load(modelname+'_Xstandard_dev.pkl')


# Load the spectrum restriction
os.chdir(maindirectory)
os.chdir(varsel_directory)

newpositions_optimal = joblib.load(spectrum+'_'+transform+'_'+varsel+'_newpositions_optimal.pkl')
newEEPs_optimal = joblib.load(spectrum+'_'+transform+'_'+varsel+'_EEPs_optimal.pkl') 

os.chdir(maindirectory)

# Restrict the dataset and show it to the user

# X = X
# X += 1
# X = np.log(X)
Xrest_ori = X[newEEPs_optimal]

X2plot = X_ori.iloc[sample2plot]
X2plotrest = Xrest_ori.iloc[sample2plot]

if spectrum == "AbsScan":
    
    # Plot the abs scan data
    varxID = X_ori.columns[-500:]
    waves = list(range(300,800))
    EEMpositions = waves 
    plt.plot(waves,X2plot[-500:],c="grey")
    plt.scatter(newpositions_optimal,X2plotrest)
    plt.show()
    
    # # Plot all spectra
    # for sample in obsID:
    #     plt.plot(waves,X.loc[sample,varxID])
    #     plt.title(sample)
    #     plt.ylim([0,2])
    #     fig = plt.gcf()
    #     # fig.savefig(sample,dpi=150,bbox_inches='tight')
    #     plt.show()
    
elif spectrum == "Fluoro2D": 

    # Or plot the fluoro spectral data
    varxID = X_ori.columns[-6103:]
    rcParams['figure.figsize'] = (6, 5)
    rcParams['axes.titlesize'] = (7)
    rcParams['axes.titleweight'] = (1000)
    
    
    plotasEEM(newpositions_optimal,X2plotrest)
    plt.show()
    

    # # Plot all spectra    
    # for sample in obsID:
    #     # plotasEEM(newpositions_optimal,Xrest.loc[sample])
    #     plotasEEM(positions,X.loc[sample])
    #     plt.title(sample)
    #     fig = plt.gcf()
    #     fig.savefig(sample,dpi=300,bbox_inches='tight')
    #     plt.show()


# Pre-process the spectral data

X = X[varxID]

X_transf = datatransform_testset(X,transformation,X_mean,X_standard_dev,1)
Xrest = X_transf[newEEPs_optimal]
Xrest_ori = X[newEEPs_optimal]

if MLalg == 'CNNR_2D':
    Xrest= reconstructEEMdatabase(Xrest.to_numpy(),newpositions_optimal)
    y_pred = model_optimal.predict(Xrest)
else:
    y_pred = model_optimal.predict(Xrest.to_numpy())

if transformation == 'LogBoth' or transformation == 'LogOutput':
    y_pred = math.e**y_pred - 1
    

# Use ML model for deriving a process parameter and save predictions to .xlsx

output_values = pd.DataFrame(data=y_pred,
                         index = obsID,
                         columns = [output2predict+" - Model"])

output_values.to_excel(name4results)


# Plot spectra with result of output on the title

if plot_results_withspectra:

    os.chdir(folder4figures)
    
    if spectrum == "Fluoro2D":
        for sample in obsID:
            plotasEEM(newpositions_optimal,Xrest_ori.loc[sample])
            # plotasEEM(positions,X.loc[sample])
            plt.title(sample+": "+ output_name+ " "+str(np.round(output_values.loc[sample][0],2)))
            fig = plt.gcf()
            fig.savefig(sample+": "+ output_name+ " "+MLalg,dpi=300,bbox_inches='tight')
            plt.show()
            
    elif spectrum == "AbsScan":    
        for sample in obsID:
            plt.plot(waves,X.loc[sample,varxID],c='grey')
            plt.scatter(newpositions_optimal,Xrest_ori.loc[sample,newEEPs_optimal])
            plt.title(sample+": "+ output_name+ " "+str(np.round(output_values.loc[sample][0],2)))
            plt.ylim([0,2.5])
            fig = plt.gcf()
            fig.savefig(sample+": "+ output_name+ " "+"_absML",dpi=150,bbox_inches='tight')
            plt.show()
    
    
    os.chdir(maindirectory)

# Plot accuracy if validation data is given

from aux_functions import plotAccScatters

if validation_data:

    monitordata = pd.read_excel(validation_data).set_index('Obs_ID')
    output = monitordata.join(output_values).dropna()
    # plt.scatter(output[output2predict],output[output2predict+" - Model"])
    plotAccScatters(output,output2predict)
    




