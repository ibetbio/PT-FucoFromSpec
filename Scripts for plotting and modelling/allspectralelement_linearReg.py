# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 18:53:08 2022

@author: Pedro

Script for computing linear regression for all spectral wavelengths

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from aux_functions_2 import r2,rmse,maxk
from aux_functions_2 import collectdata
from matplotlib import rcParams
from sklearn.linear_model import LinearRegression
import math
from aux_functions import datasetget_group

from aux_functions import plotAccScatters,plotAccScatters_timewise
from aux_functions import linearcalibration

SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
Trainsub = str.maketrans("t","ₜ")


# Collect the data

folder = "C:/Users/pedro.brandao/OneDrive - iBET/Documents - PB/Fuco2D_May24/Suppplementary Attachments/S1/DataSet"
folder4augmenteddata = "C:/Users/pedro.brandao/OneDrive - iBET/Documents - PB/Fuco2D_May24/Suppplementary Attachments/S2.3"


monitordata,sampledata,fluorodata,absdata,absextdata,no3po4data = collectdata(folder)


monitordata_ori = monitordata.copy()
fluorodata_ori = fluorodata.copy()
badchars = ["(",")","/"," ","."]
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

# ============================ USER INPUTS ================================== #

path2savefigs = "C://Users//pedro.brandao//OneDrive - iBET//Python scripts//Learning PT_version110823//LinRegs//"

criterion = 'Main Split' #  'Assay_ID' # 

training_set =  ['Training','Training 2'] # ['PTop','PTT2'] #
validation_sets = [['Validation','Validation 2']] #  ,['Temporal Prediction']] # # [['PTT1']] 
 

analyse_robustness = True

LOBO_splitcriterion = 'Assay_ori'


valcolors = [(0.7490196078431373, 0.5647058823529412, 0),'C2','C4','C3']


metabatches2remove = []
remove_peptone = 0
batches2remove = ['PTT1_01','PTT1_05','PTT1_07','PTT1_18','PTT1_20','PTT1_21','PTT1_22',
                   'PTop_07','PTop_02']

batches2remove = ['PTT1_01','PTT1_04','PTT1_05','PTT1_10','PTT1_13','PTT1_18','PTT1_19']


obs2remove = []
assays2remove = []

transformation = 0
double_log = 0
log_shift = 1



output2predict = "CC (M/mL)"
predictors2remove = [] # ["Fx (pg/cell)","Chl (mg/g)"]

noML = 1
predictor = 'Abs at 750 nm'
fit_intercept = True

include_all_outputs = 0
include_2DF = 0
include_abs = 1
include_absext = 0

viability_tres = 0


# ========================================================================== #

data4stats = []
iteration = -1
validiter = -1

if include_2DF:
    spectrum = "_with2DF"
elif include_abs:
    spectrum = "_withAbs"
elif include_absext:
    spectrum = "_withAbsext"
elif include_absext:
    spectrum = "_withMUSE"

for val in validation_sets:
    
    validiter +=1
    
    df = pd.DataFrame()
    df = sampledata.join(monitordata[[output2predict,'Viability (%)']])
    # df = df.join(absVShplcdata)
    # df = df.join(absdata)
    # df = df.join(absextdata)   
    
    obs2remove += list(df[df['Viability (%)'] < viability_tres].index)
    
        
    batches2remove = list(dict.fromkeys(batches2remove))
    
    df = df[[not elem for elem in df.Batch_ID.isin(batches2remove)]]
    df = df[[not elem for elem in df.Assay_ID.isin(metabatches2remove)]]
    df = df[[not elem for elem in df.index.isin(obs2remove)]]
    df = df[[not elem for elem in df.Assay_ori.isin(assays2remove)]]
    df = df.drop(columns=['Viability (%)'])
    df = df.dropna()

    varxID = []
    
    sampledata = df.iloc[:,:11]

    if include_2DF:
        df = df.join(fluorodata)
        varxID += list(fluorodata.columns)
    
    if include_abs:
        df = df.join(absdata)
        varxID += list(absdata.columns)
    
    if include_absext:
        df = df.join(absextdata)
        varxID += list(absextdata.columns)
    
    varyID = list(monitordata.columns)
    

    r2_LOBO = []
    r2_LOBO_train = []
    ypred_LOBO = pd.DataFrame()
    rmseP_LOBO = []
    
    rcParams['figure.figsize'] = (3, 2.5)
    rcParams['axes.titlesize'] = (6)
    rcParams['axes.titleweight'] = (1000)
    
    iteration += 1
        
    X_ori_train,y_ori_train,obs_ID_train,batch_ID_train,df_train \
                = datasetget_group(df,varxID,output2predict,training_set,criterion)          

    X_ori_test,y_ori_test,obs_ID_test,batch_ID_test,df_test \
                = datasetget_group(df,varxID,output2predict,val,criterion)    
    
    day_ID = df['Day'].to_list()
    days = list(dict.fromkeys(day_ID))
    days.sort()    
    
    day_ID_train = df_train['Day'].to_list()
    days_train = list(dict.fromkeys(day_ID_train))
    days_train.sort()
    
    day_ID_test = df_test['Day'].to_list()
    days_test = list(dict.fromkeys(day_ID_test))
    days_test.sort()   
    
         
    ymax = 1.25*np.max(df[output2predict])
    stdy = np.std(df_train[output2predict])
    yplotmax = ymax
    
        
            
    print('Obtaining equation...')
        
    if include_all_outputs:
        predictors = varyID+varxID
    else:
        predictors = varxID
     
    for predictor2remove in predictors2remove:
        predictors.remove(predictor2remove)
    
    if include_all_outputs:
        predictors.remove(output2predict)
 
    if noML == 1:
        predictors = [predictor]
        bestpred = predictor
        
    
    df4work = df_train[predictors+[output2predict]]
    
    if iteration == 0:
        
        results_linReg = []
        
        for predictor in predictors:
            
            # print("Testing "+predictor+" as predictor")
                             
            if transformation == 0:
                x = np.array(df4work[predictor])
                y = np.array(df4work[output2predict])
                model = LinearRegression(fit_intercept=fit_intercept)
                if not fit_intercept:
                    b = 0
                
            elif transformation == 1:
                x = np.array(df4work[predictor])
                if double_log:
                    x = np.log(x+log_shift)
                y = np.array(np.log(df4work[output2predict]+log_shift))  
                model = LinearRegression(fit_intercept=True)
                
                
            x = x.reshape(-1,1)
            y = y.reshape(-1,1)
            model.fit(x,y)
            
            r_sq = model.score(x,y)
            m = model.coef_[0][0]
    
            if fit_intercept or transformation == 1:
                b = model.intercept_[0] 
            else:
                b = 0
    
            # results_linReg.append([predictor,r_sq,m,b])
            
            if analyse_robustness:
                

                allypred_LOBO = pd.DataFrame()
                rmseP_LOBO_list = []
                r2_LOBO = []     
                
                # G.1 - Initiate LOBO loop
                                
                df_LOBO = df.copy()
                
                df_LOBO = df_LOBO[df_LOBO[criterion].isin(training_set)]
                
                LOBO_splits = list(df_LOBO[LOBO_splitcriterion].unique())
                LOBO_splits.sort()
                
               
                for split in LOBO_splits:
                    
                    # G.2.1 - Get training and testing sets
                    
                    trainsplit = LOBO_splits.copy()
                    trainsplit.remove(split)
                    
                    # print(trainsplit)
                    
                    trainlist = []
                    for item in trainsplit:
                        trainlist+=[item]
          
                    split = [split]
                                          
                    X_ori_trainLOBO,y_ori_trainLOBO,obs_ID_trainLOBO,batch_ID_trainLOBO,df_trainLOBO \
                                = datasetget_group(df_LOBO,varxID,output2predict,trainlist,LOBO_splitcriterion) 
                    
                    X_ori_testLOBO,y_ori_testLOBO,obs_ID_testLOBO,batch_ID_testLOBO,df_testLOBO \
                                = datasetget_group(df_LOBO,varxID,output2predict,split,LOBO_splitcriterion)
                    
                    y_ori_testLOBO = y_ori_testLOBO[output2predict]
                                
                    # print("Split size: ",X_ori_testLOBO.shape[0])
                    
                    allypred_LOBO = pd.concat([allypred_LOBO,y_ori_testLOBO])

                
                    if transformation == 0:
                        x_LOBO = np.array(df_trainLOBO[predictor])
                        y_LOBO = np.array(df_trainLOBO[output2predict])
                        model_LOBO = LinearRegression(fit_intercept=fit_intercept)
                        
                    elif transformation == 1:
                        x_LOBO = np.array(df_trainLOBO[predictor])
                        if double_log:
                            x_LOBO = np.log(x_LOBO+log_shift)
                        y_LOBO = np.array(np.log(df_trainLOBO[output2predict]+log_shift))  
                        model_LOBO = LinearRegression(fit_intercept=True)
                   
                    
                    # Train LOBO linreg
                     
                    x_LOBO = x_LOBO.reshape(-1,1)
                    y_LOBO = y_LOBO.reshape(-1,1)
                    model_LOBO.fit(x_LOBO,y_LOBO)
                     
                    r_sq_LOBO = model_LOBO.score(x_LOBO,y_LOBO)
                    m_LOBO = model_LOBO.coef_[0][0]
                    
                    if fit_intercept or transformation == 1:
                        b_LOBO = model_LOBO.intercept_[0] 
                    else:
                        b_LOBO = 0  
                        
                    # print('m_LOBO = ',m_LOBO)
                    # print('b_LOBO = ',b_LOBO)
                     

                    
                    # Test LOBO linreg 
                    
                    df_testLOBO = sampledata.join(df_testLOBO[[predictor,output2predict]]).dropna()
                    
                    y_testLOBO = df_testLOBO[output2predict]
                    x_testLOBO = df_testLOBO[predictor]
                    
                    if transformation == 1:
                        y_testLOBO = np.log(y_testLOBO+log_shift)   
                        if double_log:
                            x_testLOBO = np.log(x_testLOBO+log_shift)
                        
                    y_predLOBO = x_testLOBO*m_LOBO + b_LOBO    
                    
                    if transformation == 1:
                        y_predLOBO = math.e**y_predLOBO-log_shift
                        y_testLOBO = math.e**y_testLOBO-log_shift

                    rmseP_LOBO = np.round(rmse(np.array(y_testLOBO),np.array(y_predLOBO)),4)
                    rsq_LOBO = np.round(r2(np.array(y_testLOBO),np.array(y_predLOBO)),4)
                    
                    # G.2.5 - Store results
                    allypred_LOBO.loc[y_testLOBO.index,'Model'] = y_predLOBO
                    rmseP_LOBO_list.append(rmseP_LOBO)
                    r2_LOBO.append(rsq_LOBO)
                    
                    
                # G.3 - Compute Q2, RMSECV
                
                allypred_LOBO = allypred_LOBO.rename(columns={0:'Experimental'})
                allypred_LOBO = allypred_LOBO.join(sampledata['Assay_ori'])
                
                q2 = r2(allypred_LOBO['Experimental'].values,allypred_LOBO['Model'].values)
                rmseCV = rmse(allypred_LOBO['Experimental'].values,allypred_LOBO['Model'].values)  
                
                
                # G.4 - Draw accuracy plots
                
                # plotAccScatter_LOBO(allypred_LOBO,['C0'],
                #                                   output2predict,yplotmax,
                #                                   stdy,q2,rmseCV,False)
                # fig = plt.gcf()
                
                path2savefigs = "C://Users//pedro.brandao//OneDrive - iBET//Python scripts//"+\
                                    "Learning PT_version110823//Results//Images//"
                                    
                output_name = output2predict
                predictor_name = predictor
                
                for i in badchars:
                    output_name= output_name.replace(i,"")
                    predictor_name= predictor_name.replace(i,"")
                    
                # fig.savefig(path2savefigs+'LOBO_LinReg_predict_'+output_name+'_from_'+predictor_name+spectrum,
                #               dpi=150,bbox_inches='tight') 


                plt.show()   
       
            results_linReg.append([predictor,r_sq,m,b,q2,rmseCV])
            print(predictor)
        
        results_linReg = pd.DataFrame(results_linReg).set_index(0).rename(columns = {1:'r2',2:'m',3:'b',4:'q2',5:'rmseCV'})
        print(results_linReg)
     
        best = results_linReg.iloc[maxk(results_linReg['q2'].to_list(),1)[1]]
        bestpred,m,b,r2p_train,bestq2,bestrmseCV = (best.name,best['m'],best['b'],np.round(best['r2'],3),best['q2'],best['rmseCV'])
        print("Best predictor is ",bestpred)
        
        # Plot the linear regression data 
     
        df4plot = sampledata.join(df_train[[bestpred,output2predict]]).dropna()  
        x = df4work[bestpred]
        y_train = df4work[output2predict]
        
        if transformation == 1:
            y_train = np.log(y_train+log_shift)
            if double_log:
                x = np.log(x+log_shift)
       
        plt.scatter(y_train,x,marker='^')
        plt.plot([m*min(x)+b,m*max(x)+b],[min(x),max(x)],color='cornflowerblue',
                  linestyle='--',linewidth=1.25)
        plt.ylabel(bestpred)
        plt.xlabel(output2predict)
    
        if transformation == 1:
            plt.xlabel('Log '+output2predict)
            if double_log:
                plt.ylabel('Log '+bestpred)
        
        fig1 = plt.gcf()
        plt.show()
         
       
        if transformation == 0:
            y_predtrain = x*m+b
            
        else:
            y_predtrain = x*m+b
            y_train = math.e**y_train-log_shift
            y_predtrain = math.e**y_predtrain-log_shift
            
        rmseT = rmse(y_train.to_numpy(),y_predtrain.to_numpy())
        r2_train = r2(y_train.to_numpy(),y_predtrain.to_numpy())
    
        fig2,ax = plt.subplots(figsize=(2, 1))
        ax.set_axis_off()
        
        if transformation == 0:
            plt.text(0,0,output2predict+" = "+str(np.round(m,2))+' x '+bestpred+" + "+str(np.round(b,2)))      
        elif double_log:
            plt.text(0,0,"Log " + output2predict+" = "+str(np.round(m,2))+' x Log '+bestpred+" + "+str(np.round(b,2)))
        else:
            plt.text(0,0,"Log " + output2predict+" = "+str(np.round(m,2))+' x '+bestpred+" + "+str(np.round(b,2)))
            
        plt.show()

    
    # Testing
            
    df4work_test = sampledata.join(df_test[[bestpred,output2predict]]).dropna() 
    y_test = df4work_test[output2predict]
    x = df4work_test[bestpred]
    
    if transformation == 1:
        y_test = np.log(y_test+log_shift)   
        if double_log:
            x = np.log(x+log_shift)
        
    y_pred = x*m + b

    df4plot = df4work_test

    if transformation == 1:       
        y_test = math.e**y_test-log_shift
        y_pred = math.e**y_pred-log_shift
    
    y_predtrain = y_predtrain.rename(output2predict + ' - Model')
    
    y_pred = y_pred.rename(output2predict + ' - Model')
    
    r2_test = r2(y_test.to_numpy(),y_pred.to_numpy())
    rmseP = rmse(y_test.to_numpy(),y_pred.to_numpy())
  
    valcolor = valcolors[iteration]
    
    # y_train = y_train.rename({bestpred:output2predict + ' - Experimental'})
    # y_test = y_pred.rename({bestpred:output2predict + ' - Experimental'})   
    
    y_pred = pd.DataFrame(y_pred)
    y_test = pd.DataFrame(y_test)
    
    output_name = output2predict
    predictor_name = bestpred
    
    for i in badchars:
        output_name= output_name.replace(i,"")
        predictor_name= predictor_name.replace(i,"")
        
    
    fig3,fig4,fig6 = plotAccScatters(y_train,
                                y_predtrain,
                                y_test,
                                y_pred,
                                output2predict,yplotmax,stdy,valcolor,r2_train,r2_test,rmseT,rmseP,sampledata)           
    
        
    fig5 = plotAccScatters_timewise(sampledata,y_train,y_predtrain,obs_ID_train,np.array(y_pred),y_test,
                         obs_ID_test,output2predict,days,valcolor,yplotmax)
    
    path2savefigs = "C://Users//pedro.brandao//OneDrive - iBET//Python scripts//"+\
                        "Learning PT_version110823//Results//Images//"

    fig6.savefig(path2savefigs+'TrainTest_LinReg_predicting_'+output_name+'_from_'+predictor_name+spectrum,
                 dpi=150,bbox_inches='tight')    
    fig6.savefig('TrainTest_LinReg_predicting_'+output_name+'_from_'+predictor_name+spectrum,
                 dpi=150,bbox_inches='tight')   

    
            

# Plot variables scanned

from aux_functions_2 import plotasEEM,getEEMpositions2

rcParams['figure.figsize'] = (3, 2.5)

if include_2DF:
    EEMpositions,translations = getEEMpositions2(250, 790, 260, 800, 5)
    translations = translations.set_index('ids')       
    newpositions_optimal = [translations.loc[bestpred]['coords']]  

    # plotasEEM(newpositions_optimal,EEM4plot.loc[bestpred])
    result2plot = results_linReg['q2']
    result2plot[result2plot < 0] = 0
    plotasEEM(EEMpositions,np.array(results_linReg['q2']))
    plt.xlabel("Excitation Wavelenght (nm)")
    plt.ylabel("Emission Wavelenght (nm)")
    plt.show()

if include_abs:

    waves = range(300,800)
    
    plt.bar(waves,results_linReg['q2'])
    # plt.plot(waves,X_ori_train.iloc[1,2:],c="C1")
    plt.ylim([0.4,1])
    plt.xlabel("Wavelenght (nm)")
    plt.ylabel("Q2")
    

    