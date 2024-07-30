# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 14:51:55 2021

Machine Learning of Phaeodactylum tricornutum cultures: Regression

@author: Pedro

inspired in 

https://towardsdatascience.com/how-to-train-a-classification-model-with-tensorflow-in-10-minutes-fd2b7cfba86
https://www.kaggle.com/shelvigarg/wine-quality-dataset
https://github.com/EBjerrum/Deep-Chemometrics/blob/master/Deep_Chemometrics_with_data_augmentation.py.ipynb


In this cell, we import the required libraries, functions and classes, perform
character conversions so we may use them in the plots, and fetch the data of 
axenic PT cultures

"""

# Library importation

import math,os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from aux_functions_2 import r2,rmse,plotasEEM, fluoro1Dgen
from aux_functions_2 import getEEMpositions2,windowgen,windowgen_linear,autoinputselect2
from aux_functions_2 import collectdata,reconstructEEMdatabase
from matplotlib import rcParams
import joblib
from tensorflow.keras.models import load_model

from aux_functions import datasetget_group,augment_df_byday,datatransform,\
                                    obtainVIP_PLSRmodel,variselectML,computeMLmodel
from aux_functions import plotAccScatters,plotAccScatters_timewise,plotAccScatter_LOBO,\
                                    create_folder_if_not_exists,datatransform_testset
from aux_functions import linearcalibration
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform

# This is for character conversion for naming files without special characters

SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
Trainsub = str.maketrans("t","ₜ")
badchars = ["(",")","/"," ","."]

folder = "C:/Users/pedro.brandao/OneDrive - iBET/Documents - PB/Fuco2D_May24/Suppplementary Attachments/S1/DataSet"
folder4augmenteddata = "C:/Users/pedro.brandao/OneDrive - iBET/Documents - PB/Fuco2D_May24/Suppplementary Attachments/S2.3"

# Collect the data

monitordata,sampledata,fluorodata,absdata,absextdata,no3po4data = collectdata(folder)

monitordata_ori = monitordata.copy()
fluorodata_ori = fluorodata.copy()
sampledata_ori = sampledata.copy()


#%%

"""
This cell serves for performing linear calibrations between variables
and creates new variables, such as Fx (mg/g) or Chl per cell 

"""

# Keep original data saved

monitordata = monitordata_ori.copy()
sampledata = sampledata_ori.copy()
monitordata = monitordata.join(no3po4data)

# Obtain parameters from linear calibrations with standard analytical equipment

# Dryweight (in the data we have some dryweight measurements, 'DW - experimental (g/L)')

new_variable = "MUSE DryWeight"
# new_variable = "MD750"

monitordata[new_variable] = 1e-3*monitordata['CC (M/mL)'] * np.log(monitordata['CytoFSC (a.u.)']).pow(3)
# monitordata[new_variable] = absdata["Abs at 750 nm"]
fit_intercept = True
logtransform = False
samples2consider = sampledata # [sampledata['Assay_ori'].isin(['F2','F2+N','F2+N+P','F2+N+P_part2'])]

standard = monitordata['DW - experimental (g/L)'].dropna()
wave = pd.Series(monitordata['MUSE DryWeight'])
m0,b0 = linearcalibration(samples2consider,standard,wave,fit_intercept,logtransform)

new_variable = 'DW (g/L)'

monitordata[new_variable] = m0*monitordata['MUSE DryWeight'] + b0

# Fucoxanthin
            
monitordata['Fx (pg/cell)'] = monitordata['Fx (ppm)']/monitordata['CC (M/mL)']
monitordata['Fx (mg/g)'] = monitordata['Fx (ppm)']/monitordata['DW (g/L)']


# Include only the variables of interest
monitordata = monitordata[['CC (M/mL)','CytoRed (a.u.)',
                           'Fx (ppm)','Viability (%)']]



#%%

sampledata = sampledata_ori.copy()

"""
This cell is where machine learning models are computed according to 
the User's preferences 


"""

# ============================ USER PREFERENCES ============================ #

criterion = 'Main Split' # column based on which the train/test split will be done

validation_sets =  [['Validation','Validation 2']] # validation samples label

training_set = ['Training','Training 2'] # training samples label

# Generate or load augmented data set
load_aug_data = True
save_aug_data = False


# Save the models here computed
save_models = False

# Cross-validation options
analyse_robustness = False
LOBO_splitcriterion = 'Assay_ori'
LOBO_splits = 'All' 

betashift2 = 0.1 # this is to introduce the random noise 
data_augmentation_reps = 5 # augmentation product

valcolors = ['C0','C1','C2','C3'] # colors for accuracy plots

# Pipeline definition: Algorithm, Spectroscopy, Pre-process and variable selection

MLalgorithms = ['SVM'] # ,'PLSR','SVM','CNNR','CNNR_2D'] 
specID =   ["Fluoro2D"] # #AbsScan,Fluoro2D]
datapreprocess_types = ['MCSN'] #  'MCSN','Ori','MCSN',LogOutput']
varsels =  ['MW_2']  # 'MW_0','MW_1','MW_2','MW_3','VIP']

# Variable to predict

var2model = ['CC (M/mL)'] #  ['Fx (ppm)']  # 

# Decide to compute a new model or load one

load_architecture = False
load_varsel = False


# Remove observations by name or label

metabatches2remove = ['PTS1','PTS2','PTax3','S2021','S2022','T2022_noN','S2022_noN','T2021','T2022']


batches2remove = ['PTT1_01','PTT1_05','PTT1_07','PTT1_18','PTT1_20','PTT1_21','PTT1_22',
                   'PTop_07','PTop_02']

batches2remove = ['PTT1_01','PTT1_04','PTT1_05','PTT1_10','PTT1_13','PTT1_18','PTT1_19']


assays2remove = []
obs2remove = []

log_shift = 1

EEM4image = 'PTop_04_4'
show_temporal_plots = False
showVS = False

tune = True
innerCV_foldby = 'Assay_ori'

# ML challenge

MLproblem = "Regression"

# PLSR variable selection Hyperparameters

VIPstep = 1

# PLSR Hyperparameters

PLSR_hyperparameters2tune = {"n_components":   list(range(1,6,1))} 

# XGB Regression Hyperparameters

XGB_hyperparameters2tune = {
                            'n_estimators': sp_randint(50, 200),  
                            'learning_rate': sp_uniform(0.1,0.5),  
                            'max_depth': [10],
                            'subsample': sp_uniform(0.5, 1.0),
                            'colsample_bytree': sp_uniform(0.5, 1.0),
                            'gamma': sp_uniform(0, 20),
                            'min_child_weight': sp_randint(1, 20),  
                            }

# Random Forest Hyperparameters

Forest_hyperparameters2tune = { 
                                'n_estimators': sp_randint(50, 200),
                                'max_depth': sp_randint(1, 20),
                                'min_samples_split': sp_randint(2, 20),
                                'min_samples_leaf': sp_randint(1, 20),
                                'max_features': ['auto', 'sqrt', 'log2'],
                                'bootstrap': [True, False],
                                }

# SVR Hyperparameters

SVM_hyperparameters2tune = {
                            'C': sp_uniform(loc=0.1, scale=5),
                            'kernel': ['linear', 'rbf','sigmoid'],
                            'degree': sp_randint(1, 10),  
                            'gamma': ['scale', 'auto'] + list(sp_uniform(loc=0.001, scale=0.1).rvs(5)),
                            'epsilon': sp_uniform(loc=0.0, scale=0.2),  
                            'shrinking': [True, False],
                            }


# CNNR hyperparameters
visualizeCNNtrain = False

CNNR_hyperparameters2tune = {'C1_K': 32,
                        'C1_S': 3,
                        'C2_K': 64,
                        'C2_S': 3,
                        'DENSE': 128,
                        'DROPOUT': 0.5,
                        'learning_epochs': 50,
                        'learning_batchsize': 16,
                        'activation' : 'relu',
                        'rdlr': []}

# CNNR 2-D 
CNNR2D_hyperparameters2tune = {'C1_K': 32,
                                'C1_S': 3,
                                'C2_K': 64,
                                'C2_S': 3,
                                'DENSE': 128,
                                'DROPOUT': 0.5,
                                'learning_epochs': 50,
                                'learning_batchsize': 16,
                                'activation' : 'relu',
                                'rdlr': ''}



# ========================= END of USER INPUTS ============================= #


maindirectory = os.getcwd()

varyID = list(monitordata.columns)

top_win_tres = 0.98
Q2samplelables = False
viability_tres = 0

rcParams['figure.figsize'] = (3, 2.5)
rcParams['axes.titlesize'] = (7)
rcParams['axes.titleweight'] = (1000)


for output2predict in var2model: 
    
    if analyse_robustness:

        result_df = [['SpecID','DataPreProcess','MLalg','VarSel',
                     'Spectrum used (%)','R2t'.translate(SUP).translate(Trainsub),'RMSET',
                     'R\u00b2','RMSEP1','Q2'.translate(SUP),'RMSECV1',
                     'R\u00b2','RMSEP2','Q2'.translate(SUP),'RMSECV2',
                     'R\u00b2','RMSEP3','Q2'.translate(SUP),'RMSECV3',
                     'R\u00b2','RMSEP4','Q2'.translate(SUP),'RMSECV4']]
    else:
        result_df = [['SpecID','DataPreProcess','MLalg','VarSel',
                     'Spectrum used (%)','R2t'.translate(SUP).translate(Trainsub),'RMSET',
                     'R\u00b2','RMSEP1',
                     'R\u00b2','RMSEP2',
                     'R\u00b2','RMSEP3',
                     'R\u00b2','RMSEP4']]        
        
    output_name = output2predict
    
    for i in badchars:
        output_name= output_name.replace(i,"")
             
    for spectrodata2use in specID:
        
        if spectrodata2use == "AbsScan":
            spectrodata = absdata  
            EEMpositions = list(range(300,800))   
            translations = []
            
        elif spectrodata2use == "Fluoro2D":
            spectrodata = fluorodata
            EEMpositions,translations = getEEMpositions2(250, 790, 260, 800, 5)
            # EEMpositions,translations = getEEMpositions2(260, 760, 270, 770, 25)
                   
        varxID = list(spectrodata.columns)        
        
        df = pd.DataFrame()
        df = sampledata.join(monitordata[[output2predict,'Viability (%)']])
        df = df.join(spectrodata)
        df = df.join(absextdata)   
        
        obs2remove += list(df[df['Viability (%)'] < viability_tres].index)
            
        batches2remove = list(dict.fromkeys(batches2remove))
        
        df = df[[not elem for elem in df.Batch_ID.isin(batches2remove)]]
        df = df[[not elem for elem in df.Assay_ID.isin(metabatches2remove)]]
        df = df[[not elem for elem in df.index.isin(obs2remove)]]
        df = df[[not elem for elem in df.Assay_ori.isin(assays2remove)]]
        
        df = df.drop(columns=['Viability (%)'])
        df = df.dropna()
        
        sampledata = df.iloc[:,:12]
        
        
        for transformation in datapreprocess_types:
                
            for MLalg in MLalgorithms:
        
                MLalgorithms_tandem = ['PLSR', 'PLSR', MLalg]
                                        
                if MLalg == 'CNNR' or  MLalg == 'CNNR_2D':
                    # data_augmentation_reps = 5
                    # betashift2 = 0
                    visualizeCNNtrain = False
                
                if spectrodata2use == "Fluoro2D":
                    betashift = 0.0
                    slopeshift = 0.0
                    multishift = 0
                else:
                    betashift = 0 
                    slopeshift = 0 
                    multishift = 0 
                
                variable_selection = 'MW'
                minimal_numberofvariables = 50
                movingwindowtop = 1  
                dim = 1
                overlap = 0.75
                
                for varsel in varsels:  
                    
                    name4file = output_name+'_'+MLalg+'_'+spectrodata2use+'_'+transformation+'_'+varsel

                    if varsel == 'MW_0':
                        movingwindowtop = 1  
                        dim = 1
                        overlap = 0.1
                    
                    if varsel == 'MW_1':
                        movingwindowtop = 1
                        dim = 0.25
                        overlap = 0.75
                        
                    if varsel == 'MW_2':
                        dim = .125
                        overlap = 0.75
                        movingwindowtop = 10
                        
                    if varsel == 'MW_3': 
                        dim = .05
                        overlap = 0.1
                        movingwindowtop = 6
                       
                    if varsel == "VIP" or varsel == "Moving-excitation":
                        variable_selection = varsel

                    if spectrodata2use == "AbsScan" and varsel != 'MW_0':               
                        # For abs scan use a more restrict selection of variables
                        dim *= 0.5
                        overlap = overlap * 1.25
                        minimal_numberofvariables = 20
                        
                    yplotmax = 1.25*np.max(df[output2predict])                     
                         
                    X_ori = df[varxID].copy()
                    y_ori = df[['Day',output2predict]].copy() 
                    
                    obs_ID = X_ori.index
                    
                    assay_ID = df['Assay_ID'].to_list()  
                    assays =  list(dict.fromkeys(assay_ID))
                    
                    day_ID = df['Day'].to_list()  
                    days =  list(dict.fromkeys(day_ID))
                    days.sort()
                               
              
                    # Training procedure for obtaining model and its architecture
                    
                    
                    # A - Get the training data
                    
                    X_ori_train,y_ori_train,obs_ID_train,batch_ID_train,df_train \
                                = datasetget_group(df,varxID,output2predict,training_set,criterion)                       
                    print('Train set size: ',len(df_train))              
                    
                    
                    # B - Augment and transform it if required
                    
                    if load_aug_data:
                        os.chdir(folder4augmenteddata+'/'+ output_name+'augmented_data')
                        y_ori_train_aug = joblib.load(output_name+"_ori_train_aug.pkl")
                        os.chdir(maindirectory)
                    else:
                        y_ori_train_aug = augment_df_byday(y_ori_train,data_augmentation_reps,
                                                           betashift2,output2predict)
                        
                    if save_aug_data:
                        os.mkdir(output_name+'augmented_data')
                        os.chdir(output_name+'augmented_data')
                        joblib.dump(y_ori_train_aug,output_name+"_ori_train_aug.pkl")
                        os.chdir(maindirectory)
                        
                    # Repeat X data
                    
                    X_ori_train_aug = pd.concat([X_ori_train] * (data_augmentation_reps+1), ignore_index=False)   
                                 
                    # Drop the columns 'Day' and 'Assay_ori'
                    
                    X_ori_train_aug = X_ori_train_aug[varxID]
                    y_ori_train_aug = y_ori_train_aug[[output2predict]]
                    X_ori_train = X_ori_train[varxID]
                    y_ori_train = y_ori_train[[output2predict]]      
                    
                    stdy = np.std(y_ori_train)
                        
                    X_train_aug,y_train_aug,X_mean,X_standard_dev =\
                        datatransform(X_ori_train_aug,y_ori_train_aug,transformation,log_shift)
                    X_train,y_train,X_mean,X_standard_dev = datatransform(X_ori_train,y_ori_train,transformation,log_shift)
                    
                    
                    EEM4plot = X_ori.loc[EEM4image]
                    EEM4plot += 1
                    EEM4plot = np.log(EEM4plot)
                    
                          
                    if not load_architecture and not load_varsel:
                                                                                              
                        # D - Tuning an architecture: Variable Selection Machine Learning
      
                        # D.1 - Get the selection profiles according to user input
    
                        if spectrodata2use == "AbsScan" and variable_selection != 'VIP': 
                            EEPs_selection,numberofvariables = windowgen_linear(500,int(dim*499),overlap)
                                
                        if spectrodata2use == "Fluoro2D" and variable_selection != 'VIP':
                            
                            if variable_selection == 'Moving-excitation':
                                
                                varx_rest = fluoro1Dgen(varxID,range(250,790,5),4,1)  
                                # varx_rest = fluoro1Dgen(varxID,range(260,770,25),4,1) 
                                numberofvariables = [len(eeps) for eeps in varx_rest]
                                EEPs_selection = [list(translations[translations['ids'].isin(eeps)].index) for eeps in varx_rest]
                                
                            else:
                                EEPs_selection,numberofvariables = windowgen(EEMpositions,int(dim*109),overlap)
                                # EEPs_selection,numberofvariables = windowgen(EEMpositions, int(dim*24), overlap)
                                
                                from aux_functions_2 import plot_partialspectrum2
                                
                                # for window in EEPs_selection:
                                #     newEEPs = translations[translations.index.isin(window)]['ids']
                                #     plot_partialspectrum2(X_train,X_train[newEEPs], EEM4image,vmin=0,vmax=1000)
    
                        if variable_selection == 'VIP':
                            vips = obtainVIP_PLSRmodel(MLproblem,X_train_aug,y_train_aug,transformation,
                                                       PLSR_hyperparameters2tune,innerCV_foldby,sampledata)
                            
                            EEPs_selection,numberofvariables = autoinputselect2(vips,VIPstep,X_ori,
                                                                                EEMpositions,
                                                                                minimal_numberofvariables)
                            

                        
                        # D.2 - Compute ML models for each of the profiles and return the optimal ML-wise selection
                    
                        window_optima,newEEPs_optimal,newpositions_optimal = \
                                    variselectML(MLproblem,variable_selection,X_ori_train,X_train_aug,y_train_aug,X_train,y_train,transformation,
                                                 varxID,output2predict,EEPs_selection,EEMpositions,
                                                 MLalg,PLSR_hyperparameters2tune,innerCV_foldby,
                                                 showVS,spectrodata2use,EEM4plot,yplotmax,stdy,movingwindowtop,top_win_tres,log_shift,sampledata,
                                                 translations,EEM4image)
                                        
                                    
                        # Save PLSR varsel for this output
                        
                        if save_models:
          
                            create_folder_if_not_exists(output_name+"_Optimal_VarSel")
                            
                            os.chdir(output_name+"_Optimal_VarSel")
                                                      
                            joblib.dump(newEEPs_optimal,spectrodata2use+'_'+transformation+'_'+varsel+'_EEPs_optimal.pkl')
                            joblib.dump(newpositions_optimal,spectrodata2use+'_'+transformation+'_'+varsel+'_newpositions_optimal.pkl') 
                            
                        # Plot spectrum showing optimal variables selected
                        
                        # EEM4plot2 = X_train.iloc[100] 
         
                        rcParams['figure.figsize'] = (3, 2.5)
                        if spectrodata2use == "Fluoro2D":
                            # plotasEEM(newpositions_optimal,EEM4plot[newEEPs_optimal])
                            
                            plot_partialspectrum2(X_ori_train,X_ori_train[newEEPs_optimal], EEM4image,vmin=0,vmax=1000)

                        elif spectrodata2use == 'AbsScan':
                            waves = range(300,800)
                            plt.plot(waves,EEM4plot,c="grey")
                            plt.plot(waves,EEM4plot,c="grey")
                            plt.scatter(newpositions_optimal,EEM4plot[newEEPs_optimal],
                                        c="cornflowerblue",s=24,marker='o')
                            plt.ylim([min(EEM4plot)*0.95,max(EEM4plot)*1.05])

                            
                        fig4 = plt.gcf()
                        fig4.savefig(spectrodata2use+'_'+transformation+'_'+varsel,
                                      dpi=150,bbox_inches='tight')    
                        plt.show()
                        os.chdir(maindirectory)
                        

                                         
                        # D.3 - Construct final ML model optimized with architecture trained from the data
                        
                        # This final model will get the varseldata from PLSRvarsel (MWPLSR or VIP)
                        
                    if load_varsel:
                        
                        os.chdir(output_name+"_Optimal_VarSel")
                        newpositions_optimal = joblib.load(spectrodata2use+'_'+transformation+'_'+varsel+'_newpositions_optimal.pkl')
                        newEEPs_optimal = joblib.load(spectrodata2use+'_'+transformation+'_'+varsel+'_EEPs_optimal.pkl') 
                        os.chdir(maindirectory)
                        
                        # Plot spectrum showing optimal variables selected
                        rcParams['figure.figsize'] = (3, 2.5)
                        if spectrodata2use == "Fluoro2D":
                            # if transformation == "MCSN":
                            #     plotasEEMcontour(newpositions_optimal,EEM4plot[newEEPs_optimal],
                            #                      levels = [-1.5,-.75,-.1,0,.1,0.75,1.5])
                            #     # plotasEEMcontour(EEMpositions,EEM4plot,
                            #     #                  levels = [-1.5,-.75,-.1,0,.1,0.75,1.5])
                            # else:
                            plotasEEM(newpositions_optimal,EEM4plot[newEEPs_optimal])
                            
                        elif spectrodata2use == 'AbsScan':
                            waves = range(300,800)
                            plt.plot(waves,EEM4plot,c="grey")
                            plt.scatter(newpositions_optimal,EEM4plot[newEEPs_optimal],
                                        c="cornflowerblue",s=24,marker='o')
                            plt.ylim([min(EEM4plot)*0.95,max(EEM4plot)*1.05])
                            
                        plt.show()
                         
                    if not load_architecture:
                        
                        if MLalg == "PLSR" or MLalg == "PLSDA": 
                            hyperparameters2tune = PLSR_hyperparameters2tune                     
                        elif MLalg == "XGB":
                            hyperparameters2tune = XGB_hyperparameters2tune
                        elif MLalg == "RFR":    
                            hyperparameters2tune = Forest_hyperparameters2tune
                        elif MLalg == "SVM":
                            hyperparameters2tune = SVM_hyperparameters2tune
                        elif MLalg == "CNNR":
                            hyperparameters2tune = CNNR_hyperparameters2tune
                        elif MLalg == "CNNR_2D":
                            hyperparameters2tune = CNNR2D_hyperparameters2tune
                        
                        
                        
                        model_optimal,Xrest_train,Xrest_train_aug,hyperparameters_tuned = \
                            computeMLmodel(MLproblem,X_train_aug,X_train,y_train_aug,transformation,
                                           newpositions_optimal,newEEPs_optimal,MLalg,output2predict,
                                           hyperparameters2tune,innerCV_foldby,sampledata,tune,visualizeCNNtrain)
                            
                        
                         
                            
                        # D.4 - Compute training performance      
                                
                        y_predtrain = model_optimal.predict(Xrest_train)
     
                        if transformation == 'LogBoth' or transformation == 'LogOutput':
                            y_train = math.e**y_train-log_shift
                            
                            y_predtrain = math.e**model_optimal.predict(Xrest_train)-log_shift       
                            
                        if MLproblem == 'Classification':   
                            y_predtrain = y_predtrain[:,0]   
                            y_predtrain = np.array([1 if x > 0.5 else 0 for x in y_predtrain])
                        
                        rmseT = np.round(rmse(y_train,y_predtrain),5)
                        rsq_train = np.round(r2(y_train,y_predtrain),5)
                        
                        results = [spectrodata2use,transformation,
                                   MLalg,varsel,np.round(len(newEEPs_optimal)/len(varxID)*100,0),
                                   np.round(rsq_train,4),np.round(rmseT,5)]
                
                        # D.5 - Save the model and architecture
                        
                        if save_models:
 
                            create_folder_if_not_exists(output_name+'_'+MLalg)
                            
                            os.chdir(output_name+'_'+MLalg)
                            
                            if MLalg == 'CNNR' or MLalg == 'CNNR_2D':
                                model_optimal.save(name4file+'_model_optimal')
                            else:
                                joblib.dump(model_optimal, name4file+'_model_optimal.pkl')
                                
                            joblib.dump(transformation, name4file+'_transformation.pkl')  
                            joblib.dump(y_train,name4file+'_traindata.pkl')
                            joblib.dump(y_predtrain,name4file+'_trainpred.pkl')
                            joblib.dump(hyperparameters_tuned,name4file+'_hyperparams_tuned.pkl')
                            joblib.dump(rsq_train,name4file+'_rsq_train.pkl')
                            joblib.dump(rmseT,name4file+'_rmseT.pkl')
                            joblib.dump(X_mean,name4file+'_Xmean.pkl')
                            joblib.dump(X_standard_dev,name4file+'_Xstandard_dev.pkl')
                            os.chdir(maindirectory)
                            
                            print("Hyperparameters_tuned: ",hyperparameters_tuned)
                   
                        
                    
                    # E - Validation of the model on left out data set
                   
                    # E.1 - Load the model and architecture if necessary
                    
                    else:
                        
                        os.chdir(output_name+'_'+MLalg)
                        
                        if MLalg == 'CNNR' or MLalg == 'CNNR_2D':
                            model_optimal = load_model(name4file+'_model_optimal')
                        else:
                            model_optimal = joblib.load(name4file+'_model_optimal.pkl')
                        
                        transformation= joblib.load(name4file+'_transformation.pkl')
                        y_train = joblib.load(name4file+'_traindata.pkl')
                        y_predtrain = joblib.load(name4file+'_trainpred.pkl')
                        hyperparameters_tuned = joblib.load(name4file+'_hyperparams_tuned.pkl')
                        rsq_train = joblib.load(name4file+'_rsq_train.pkl')
                        rmseT = joblib.load(name4file+'_rmseT.pkl')
                        X_mean = joblib.load(name4file+'_Xmean.pkl')
                        X_standard_dev = joblib.load(name4file+'_Xstandard_dev.pkl')
                    
                        os.chdir(maindirectory)
                    
                    

                    
                    
                    # E.2 - Get the validation/prediction data and transform it if required
                    
                    validiter = -1
                    
                    for validation_set in validation_sets:
                        
                        validiter += 1
                        valcolor = valcolors[validiter]
                        
                        X_ori_test,y_ori_test,obs_ID_test,batch_ID_test,df_test \
                                    = datasetget_group(df,varxID,output2predict,validation_set,criterion)
                        
                        y_ori_test = y_ori_test[output2predict]
                        
                        X_ori_test = X_ori_test[varxID]
                        
                        print('Test set size: ',len(df_test))  
     
                        X_test,y_test = datatransform_testset(X_ori_test,y_ori_test,transformation,X_mean,X_standard_dev,log_shift)
                        Xrest_test = X_test[newEEPs_optimal]
                                  
                        if MLalg == "CNNR_2D":
                            Xrest_test = reconstructEEMdatabase(Xrest_test.to_numpy(),newpositions_optimal)
                        
                        # E.3 - Use the trained model to make a prediction and compute accuracy
                        
                        y_pred = model_optimal.predict(Xrest_test)
                        
                        if MLproblem == 'Classification':
                            y_pred = y_pred[:,0]   
                            y_pred =np.array([1 if x > 0.5 else 0 for x in y_pred])
                            from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
                            
                            print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')
                            print(f'Precision: {precision_score(y_test,y_pred):.2f}')
                            print(f'Recall: {recall_score(y_test, y_pred):.2f}')
                            
                            print(confusion_matrix(y_test, y_pred))
                        
               
                        if transformation == 'LogBoth' or transformation == 'LogOutput':
                            y_pred = math.e**y_pred-log_shift
                            y_test = math.e**y_test-log_shift
    
                        rmseP = np.round(rmse(y_test,y_pred),4)
                        rsq = np.round(r2(y_test,y_pred),4)
                        
                        if not load_architecture:
                            results += [np.round(rsq,4),np.round(rmseP,5)]
    
                 
                        # F - Graphical accuracy figures
                        
                        # F.1 - Accuracy scatter: Experimental VS Predicted
                        
                        # stdy = np.std(y_ori_test)
                        
                        y_pred = pd.DataFrame(data = y_pred, index = y_test.index,
                                              columns = [output2predict + ' - Model'])
                        
                        y_test = pd.DataFrame(y_test)
                        
                        yplotmax = 1.25*np.max(df_test[output2predict]) 
                        
                        # yplotmax = 18
                        
                        fig7,fig8,fig9 = plotAccScatters(y_train,y_predtrain,y_test,y_pred,output2predict,
                                        yplotmax,stdy,valcolor,rsq_train,rsq,rmseT,rmseP,sampledata)
                        
                        create_folder_if_not_exists(output_name+'_'+MLalg)
                        
                        os.chdir(output_name+'_'+MLalg)
                        
                        fig9.savefig('TrainTest_'+MLalg+'_'+spectrodata2use+'_'+transformation+'_'+varsel,
                                     dpi=150,bbox_inches='tight')    
                        
                        os.chdir(maindirectory)
                        
                        # F.2 - Scatters of time-wise accuracy, if required
                                                                
                        if show_temporal_plots:
                            
                            fig8 = plotAccScatters_timewise(sampledata,y_train,y_predtrain,obs_ID_train,
                                                            np.array(y_pred),y_test,
                                                 obs_ID_test,output2predict,days,valcolor,yplotmax)
                        
                        # G - Robustness Analysis: Leave-One-Batch-Out
                        
                        if analyse_robustness:
                        
                            allypred_LOBO = pd.DataFrame()
                            rmseP_LOBO = []
                            r2_LOBO = []     
                            
                            # G.1 - Initiate LOBO loop
                            
                            # if validiter>0:
                            # LOBO_splitcriterion = 'LODO Split'
                            # else:
                            # LOBO_splitcriterion = 'LOBO Split'
                                
                            # LOBO_splitcriterion = 'LODO Split'
                            
                            df_LOBO = df.copy()
                            
                            df_LOBO = df_LOBO[df_LOBO[criterion].isin(training_set)]
                            
                            if LOBO_splits == 'All':
                                LOBO_splits = list(df_LOBO[LOBO_splitcriterion].unique())
                                LOBO_splits.sort()
                            
                            
                            for split in LOBO_splits:

                                # G.2.1 - Get training and testing sets
                                
                                trainsplit = LOBO_splits.copy()
                                trainsplit.remove(split)
                                
                                print(trainsplit)
                                
                                trainlist = []
                                for item in trainsplit:
                                    trainlist+=[item]
      
                                split = [split]
                                
                                X_ori_trainLOBO,y_ori_trainLOBO,obs_ID_trainLOBO,batch_ID_trainLOBO,df_trainLOBO \
                                            = datasetget_group(df_LOBO,varxID,output2predict,trainlist,LOBO_splitcriterion) 
                                
                                X_ori_testLOBO,y_ori_testLOBO,obs_ID_testLOBO,batch_ID_testLOBO,df_testLOBO \
                                            = datasetget_group(df_LOBO,varxID,output2predict,split,LOBO_splitcriterion)
                                
                                y_ori_testLOBO = y_ori_testLOBO[output2predict]
                                            
                                print("Split size: ",X_ori_testLOBO.shape[0])
                                
                                allypred_LOBO = pd.concat([allypred_LOBO,y_ori_testLOBO])
                                
                                # G.2.2 - Augment and transform it if required, and restrict it
                                
                                if load_aug_data:
                                    os.chdir(output_name+'augmented_data')
                                    y_ori_trainLOBO_aug = joblib.load(output_name+"_"+split[0]+"_ori_train_aug.pkl")
                                    os.chdir(maindirectory)
                                else:      
                                    y_ori_trainLOBO_aug = augment_df_byday(y_ori_trainLOBO,data_augmentation_reps,
                                                                        betashift2,output2predict)
                                    
                                if save_aug_data:
                                    os.chdir(output_name+'augmented_data')
                                    joblib.dump(y_ori_trainLOBO_aug,output_name+"_"+split[0]+"_ori_train_aug.pkl")
                                    os.chdir(maindirectory)
                                                                                                          
                                # X_ori_trainLOBO_aug = augment_df_byday(X_ori_trainLOBO,data_augmentation_reps,
                                #                                     0,varxID[0])
                                
                                X_ori_trainLOBO_aug = pd.concat([X_ori_trainLOBO] * (data_augmentation_reps+1), ignore_index=False)   
                                # Drop the column 'Day'
                                
                                X_ori_trainLOBO_aug = X_ori_trainLOBO_aug[varxID]
                                y_ori_trainLOBO_aug = y_ori_trainLOBO_aug[[output2predict]]
                                
                                X_ori_trainLOBO = X_ori_trainLOBO[varxID]
                                y_ori_trainLOBO = y_ori_trainLOBO[[output2predict]   ]                     
                                    
                                X_trainLOBO_aug,y_trainLOBO_aug,X_meanLOBO,X_standard_devLOBO =\
                                    datatransform(X_ori_trainLOBO_aug,y_ori_trainLOBO_aug,transformation,log_shift)         
                                    
                                X_trainLOBO,y_trainLOBO,X_meanLOBO,X_standard_devLOBO = datatransform(X_ori_trainLOBO,y_ori_trainLOBO,transformation,log_shift)
                                
                                X_ori_testLOBO = X_ori_testLOBO[varxID]
                                
                                X_testLOBO,y_testLOBO = datatransform_testset(X_ori_testLOBO,y_ori_testLOBO,transformation,X_meanLOBO,X_standard_devLOBO,log_shift)
                                Xrest_testLOBO = X_testLOBO[newEEPs_optimal]                           
                                
                                if MLalg == "CNNR_2D":
                                    Xrest_testLOBO = reconstructEEMdatabase(Xrest_testLOBO.to_numpy(),newpositions_optimal)
                                
                                # G.2.3 - Train model using the same hyperparams
                                
                                model_optimalLOBO,Xrest_trainLOBO,Xrest_trainLOBO_aug,hyperparameters_tuned = \
                                    computeMLmodel(MLproblem,X_trainLOBO_aug,X_trainLOBO,y_trainLOBO_aug,transformation,
                                                   newpositions_optimal,newEEPs_optimal,MLalg,output2predict,
                                                   hyperparameters_tuned,innerCV_foldby,sampledata,0,visualizeCNNtrain)
                                
                                # G.2.4 - Make the test
               
                                y_predLOBO = model_optimalLOBO.predict(Xrest_testLOBO)
                                
                                if transformation == 'LogBoth' or transformation == 'LogOutput':
                                    y_predLOBO = math.e**y_predLOBO-log_shift
                                    y_testLOBO = math.e**y_testLOBO-log_shift
                                    
                
                                rmsePsplit = np.round(rmse(y_testLOBO,y_predLOBO),4)
                                rsq_split = np.round(r2(y_testLOBO,y_predLOBO),4)
                                
                                # G.2.5 - Store results
                                allypred_LOBO.loc[y_testLOBO.index,'Model'] = y_predLOBO.reshape(len(y_predLOBO),)
                                rmseP_LOBO.append(rmsePsplit)
                                r2_LOBO.append(rsq_split)
                                
                            
                            # G.3 - Compute Q2, RMSECV
                            
                            allypred_LOBO = allypred_LOBO.rename(columns={0:'Experimental'})
                            
                            allypred_LOBO = allypred_LOBO.join(sampledata['Assay_ori'])
                            
                            q2 = r2(allypred_LOBO['Experimental'].values,allypred_LOBO['Model'].values)
                            rmseCV = rmse(allypred_LOBO['Experimental'].values,allypred_LOBO['Model'].values)  
                            
                            if not load_architecture:
                                results += [np.round(q2,4),np.round(rmseCV,5)]               
                            
                            # G.4 - Draw accuracy plots
                            
                            plotAccScatter_LOBO(allypred_LOBO,valcolor,
                                                              output2predict,yplotmax,
                                                              stdy,q2,rmseCV,Q2samplelables)
                            
                            # plotAccScatter_LOBO(allypred_LOBO,valcolor,
                            #                                   output2predict,15,
                            #                                   stdy,q2,rmseCV,Q2samplelables)
                            
                            create_folder_if_not_exists(output_name+'_'+MLalg)
                            
                            os.chdir(output_name+'_'+MLalg)
                            
                            fig = plt.gcf()
                            fig.savefig('LOBO_'+MLalg+'_'+spectrodata2use+'_'+transformation+'_'+varsel,
                                        dpi=150,bbox_inches='tight')
                            plt.show()
                            os.chdir(maindirectory)

                    # Save results
                    if not load_architecture:
                        result_df.append(results)
                        
    if not load_architecture:                
        # os.chdir("Results")                   
        result_df = pd.DataFrame(result_df)
        result_df.columns = result_df.iloc[0,:]
        result_df = result_df.drop(result_df.index[0])
        
        result_df.to_excel("results"+spectrodata2use+'_'+MLalg+'_'+output_name+'.xlsx')
        # os.chdir(maindirectory)

