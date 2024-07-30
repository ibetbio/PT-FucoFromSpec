# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 13:55:15 2023

@author: Pedro

Script for applying PCA

"""

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

import numpy as np
from aux_functions_2 import collectdata,plotasEEMcontour,getEEMpositions2,collectdata
from sklearn.linear_model import LinearRegression
import joblib
from matplotlib import rcParams

from aux_functions import DWlinearcalibration,linearcalibration

import math

# Collect the data

folder = "C:/Users/pedro.brandao/OneDrive - iBET/Documents - PB/Fuco2D_May24/Suppplementary Attachments/S1/DataSet"
folder4augmenteddata = "C:/Users/pedro.brandao/OneDrive - iBET/Documents - PB/Fuco2D_May24/Suppplementary Attachments/S2.3"


# Collect the data

monitordata,sampledata,fluorodata,absdata,absextdata,no3po4data = collectdata(folder)

monitordata_ori = monitordata.copy()
fluorodata_ori = fluorodata.copy()
sampledata_ori = sampledata.copy()

badchars = ["(",")","/"," ","."]
SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
Trainsub = str.maketrans("t","ₜ")


#%%

"""
This cell serves for performing linear calibrations between variables
and creates new variables, such as Fx (mg/g) or Chl per cell 

"""

monitordata = monitordata_ori.copy()
sampledata = sampledata_ori.copy()
monitordata = monitordata.join(no3po4data)
# Obtain parameters from linear calibrations with standard analytical equipment

monitordata['MUSE DW'] = 1e-3*monitordata['CC (M/mL)'] * np.log(monitordata['CytoFSC (a.u.)']).pow(3)
monitordata['MUSE DW'] = absdata["Abs at 750 nm"]

fit_intercept = True
logtransform = False
samples2consider = sampledata # [sampledata['Assay_ori'].isin(['F2','F2+N','F2+N+P','F2+N+P_part2','Fed F2'])]

samples2consider = sampledata[sampledata['Assay_ID'].isin(['PTop','PTT1','PTT2'])]

standard = monitordata['DW - experimental (g/L)'].dropna()
wave = pd.Series(monitordata['MUSE DW'])
m0,b0 = linearcalibration(samples2consider,standard,wave,fit_intercept,logtransform)
monitordata['DW (g/L)'] = m0*monitordata['MUSE DW'] + b0

standard = monitordata['Fx (ppm)']
wave = absextdata['ext_445']
m1,b1 = linearcalibration(samples2consider,standard,wave,fit_intercept,logtransform)
monitordata['Fx (ppm)'] = m1*absextdata['ext_445'] + b1   

            
monitordata['Fx (pg/cell)'] = monitordata['Fx (ppm)']/monitordata['CC (M/mL)']
monitordata['Fx (mg/g)'] = monitordata['Fx (ppm)']/monitordata['DW (g/L)']


import numpy as np

# Create a new series for the combined health status, initialized with zeros
cult_health = pd.Series(1, index=monitordata.index,name="Culture Health")

# Apply conditions to set the appropriate health status
cult_health[(monitordata['CytoRed (a.u.)'] >= 1000) & (monitordata['CC (M/mL)'] >= 1)] = 2
cult_health[((monitordata['CytoRed (a.u.)'] < 500) | (monitordata['CC (M/mL)'] < 1)) & (sampledata['Day'] > 3)] = 0
cult_health[(monitordata['CC (M/mL)'] < 2.5) & (sampledata['Day'] > 5)] = 0

sampledata = sampledata.join(cult_health)
monitordata = monitordata[['CC (M/mL)','CytoRed (a.u.)','Fx (ppm)']]

#%%

# ============================== USER EDIT ================================ #


# path2savefigs = 'C://Users//pedro.brandao//OneDrive - iBET//Imagens//PTfuco2D//'
# path2savefigs = 'C://Users//Pedro//OneDrive - iBET//Imagens//PTfuco2D//'

path2savefigs = 'C://Users//pedro.brandao//OneDrive - iBET//General//02. Papers Ongoing//06. PT_Crash//PCA_figures//'

criterion = 'Assay_ori'
data2use =  ['F2','F2+N','F2+N+P','F2+N+P_part2']

# assays2use = ['Training A1','Training B','Training C1','Validation A1',
#               'Validation B','Validation C1','Validation D','Validation E',
#               'Validation A2','Validation C2']

discrimination = "Assay_ori"
discriminants = ['F2','F2+N','F2+N+P','F2+N+P_part2']


viability_tres = 0

days2use = 'all' #  [3,5,7,9] #  

labels = "Obs_ID"
label_density = 10000
include_DAI = False

data_of_interest = "Standard" # "AbsScan" # "Fluoro2D"

metabatches2remove = ['PTS1','PTS2',
                        # 'T2022','S2022',
                        # 'S2022_noN','T2022_noN',
                       # 'S2021','T2021'
                      ]


batches2remove = ['PTT1_01','PTT1_05','PTT1_07','PTT1_18','PTT1_20','PTT1_21','PTT1_22',
                  'PTop_05','PTop_02']

assays2remove = []

obs2remove = [] 


vars2remove = []

PCs2plot = [1,2]

preprocess = True
negate_PC1 = False
negate_PC2 = False
use_varsel = False

biplot_loadingvectors = 6103

plot2loadingsonly = False

put_loading_labels = True
showarrows = True
biplot_loadingscaler = 1.4
arrow_w = 0.1
arrow_body_w = arrow_w/10

EEM4image = 'PTop_10_4'

figsize = (4,3.5)
figsize = (8,7)

savefig = True

# ========================================================================== #



colors = [(1.0, 0.8352941176470589, 0.4745098039215686),
                  (0.7490196078431373, 0.5647058823529412, 0),
                  (0.4980392156862745, 0.3764705882352941, 0),
                  (0.34901960784313724, 0.2627450980392157, 0),'C4']

# colors = ['red','yellow','green']


markers =  ['o','s','^','d','x','*'] # 

bigloaders = []

# Filtrate data by DAI and assay

if data_of_interest == 'Fluoro2D':
    varyID = list(fluorodata.columns)
    
    data = sampledata.join(fluorodata)
    
    fluorodatalog = fluorodata.copy()
    fluorodatalog += 1
    fluorodatalog = np.log(fluorodatalog) 
    
    # data = sampledata.join(fluorodatalog)
    put_loading_labels = False
    EEM4plot = fluorodatalog.loc[EEM4image] 
    
elif data_of_interest == 'AbsScan':
    varyID = list(absdata.columns)
    
    data = sampledata.join(absdata)
    
    absdatalog = absdata.copy()
    absdatalog += 1
    absdatalog = np.log(absdatalog)      
    
    # data = sampledata.join(absdatalog)
    
    EEM4plot = absdatalog.loc[EEM4image] 
    
elif data_of_interest == 'AbsScan2ext':
    varyID = list(absextdata.columns)
    data = sampledata.join(absextdata)
    EEM4plot = absextdata.loc[EEM4image] 
    
elif data_of_interest == 'Standard': 
    varyID = list(monitordata.columns)
    for vary in vars2remove:
        varyID.remove(vary)
    data = sampledata.join(monitordata[varyID])

# data = data.join(absdata)
# varyID += list(absdata.columns)
    
data = data[data[criterion].isin(data2use)].dropna()

# data = data[data["Main Split"].isin(data2use_2)].dropna() 

# obs2remove_aux = list(monitordata[monitordata['Viability (%)'] < viability_tres].index)



# if len(obs2remove_aux) > 0:
#     batches2remove += sampledata.loc[obs2remove_aux,:].Batch_ID

# Remove observations 


data = data[[not elem for elem in data.index.isin(obs2remove)]]
data = data[[not elem for elem in data.Batch_ID.isin(batches2remove)]]
data = data[[not elem for elem in data.Assay_ID.isin(metabatches2remove)]]
data = data[[not elem for elem in data.Assay_ori.isin(assays2remove)]]

data = data.sort_values(by='Day')


if days2use != 'all':
    data_raw = data[data['Day'].isin(days2use)]
else:
    data_raw = data

if use_varsel:
    EEPs_optimal = joblib.load('EEPs_optimal.pkl')
    newpositions_optimal = joblib.load('newpositions_optimal.pkl')
    varyID = EEPs_optimal



# discriminants = list(data_raw[discrimination].unique())

# discriminants.sort()

if include_DAI:
   varyID += ['Day']

data_raw = data_raw[varyID]     

print(varyID)


if preprocess:
    
    scaler = StandardScaler()
    scaler.fit(data_raw)
    data_raw = pd.DataFrame(index = data_raw.index,
                            columns = data_raw.columns,
                            data = scaler.transform(data_raw))
else:
    data_raw = pd.DataFrame(index = data_raw.index,
                            columns = data_raw.columns,
                            data = data_raw)        

# Perform PCA using scikit-learn

PCA_numberofcomps = PCs2plot[1] # min(len(data_raw.index)-1,len(EEPs_optimal))

pca = PCA(PCA_numberofcomps)
pca.fit(data_raw)

dataready2plot = pd.DataFrame(pca.transform(data_raw)[:,[PCs2plot[0]-1,PCs2plot[1]-1]],
                              index=data_raw.index,columns=['PC'+str(PCs2plot[0]),'PC'+str(PCs2plot[1])])
dataready2plot = sampledata.join(dataready2plot).dropna()

dataready2plot = dataready2plot.sort_values(by='Day')

if labels == "Obs_ID":
    labels = dataready2plot.index
else:
    labels = dataready2plot[labels]


count_i = 0

loadings = pd.DataFrame(data=pca.components_[[PCs2plot[0]-1,PCs2plot[1]-1],:],
                        columns=varyID,index=['PC'+str(PCs2plot[0]),'PC'+str(PCs2plot[1])])

if negate_PC1:
    dataready2plot[['PC'+str(PCs2plot[0])]] =  -dataready2plot[['PC'+str(PCs2plot[0])]]
    loadings.loc['PC'+str(PCs2plot[0])] =  -loadings.loc['PC'+str(PCs2plot[0])]
if negate_PC2:
    dataready2plot[['PC'+str(PCs2plot[1])]] =  -dataready2plot[['PC'+str(PCs2plot[1])]]
    loadings.loc['PC'+str(PCs2plot[1])] =  -loadings.loc['PC'+str(PCs2plot[1])]


    
itercolor1 = 0
fig, ax = plt.subplots(figsize=figsize)

for assay in discriminants:

    itercolor2 = 0
    data2plot = dataready2plot[dataready2plot[discrimination].isin([assay])]

    itercolor2 = 0
    
    for data in data2use:
    
        data2plot2 = data2plot[data2plot[criterion].isin([data])]
        
        print(len(data2plot2))
        
        ax.scatter(data2plot2['PC'+str(PCs2plot[0])], data2plot2['PC'+str(PCs2plot[1])],
                   marker = markers[itercolor2], color = colors[itercolor1])
    
        # plt.ylim([-10,10])
        itercolor2 += 1
    
    itercolor1 += 1
    ax.legend(data2use)
    # ax.legend(discriminants)

iterator = 0    
for i,obs in enumerate(labels):
    iterator +=1
    if np.remainder(iterator,label_density) == 0:
        ax.annotate(obs,(dataready2plot['PC'+str(PCs2plot[0])][i],
                         dataready2plot['PC'+str(PCs2plot[1])][i]),
                    fontsize = 9)
        
# Plotting according to biggest loading vector size

if not plot2loadingsonly:
    
    loadings.loc['weights',:] = 0.01*pca.explained_variance_ratio_[PCs2plot[0]-1]*loadings.loc['PC'+str(PCs2plot[0]),:].pow(2)\
        + 0.01*pca.explained_variance_ratio_[PCs2plot[1]-1]*loadings.loc['PC'+str(PCs2plot[1]),:].pow(2)
        
    loadings = loadings.T.sort_values(by="PC"+str(PCs2plot[0]),ascending=False)
    
    if showarrows:
        for i, var in enumerate(loadings.index):
            print(var)
            ax.arrow(0, 0,
                      biplot_loadingscaler*loadings.iloc[i, 0],
                      biplot_loadingscaler*loadings.iloc[i, 1],
                       head_width = arrow_w, head_length= arrow_w*1.5,
                        fc='lightsteelblue', ec='lightsteelblue')
            
            if put_loading_labels:
                ax.text(biplot_loadingscaler*loadings.iloc[i,0]*1.2,
                        biplot_loadingscaler*loadings.iloc[i,1]*1.2,
                        var, ha='center', va='center',)
            count_i += 1
            # print(var+'\n',loadings.iloc[i,:])
            
            bigloaders.append(var)
            
            if count_i == biplot_loadingvectors:
                break

else:
    
# Plottin according to the biggest contributors to PCs 1 and 2

    loadings_max_1 = list(loadings.T.pow(2).idxmax())
    loadings_max_2 = list(loadings.drop(loadings_max_1,axis = 1).T.pow(2).idxmax())
    # loadings_max = loadings_max_1 + loadings_max_2
    
    loadings_max = loadings_max_1
    for loading in loadings_max:
        if showarrows:
            ax.arrow(0, 0,
                      biplot_loadingscaler*loadings.T.loc[loading][0],
                      biplot_loadingscaler*loadings.T.loc[loading][1],
                        head_width = arrow_w, head_length= arrow_w*1.5,
                        width = arrow_body_w,
                        fc='lightsteelblue', ec='lightsteelblue')
            if put_loading_labels:
                ax.text(biplot_loadingscaler*loadings.T.loc[loading][0]*1.05,
                        biplot_loadingscaler*loadings.T.loc[loading][1]*1.05,
                        loading, ha='center', va='center',)
            count_i += 1
            
ax.axhline(y=0, color='grey')
ax.axvline(x=0, color='grey')
ax.grid(True, which='both')
plt.xlabel('PC'+str(PCs2plot[0])+" ({:.2%} explained variance)".format(pca.explained_variance_ratio_[PCs2plot[0]-1]))
plt.ylabel('PC'+str(PCs2plot[1])+" ({:.2%} explained variance)".format(pca.explained_variance_ratio_[PCs2plot[1]-1]))
plt.title("PCA Score Plot: "+data_of_interest)
# plt.xlim([-160,100])
# plt.ylim([-80,140])

fig = plt.gcf()


if savefig:

    fig.savefig(path2savefigs+'PCA_'+data_of_interest+'.png'
                  ,dpi=450,bbox_inches='tight')   
    
    fig.savefig(path2savefigs+'PCA_'+data_of_interest+'.svg'
                  ,format='svg',dpi=300)   
    
plt.show()




if data_of_interest == "Fluoro2D":
    
    for pc in ['PC'+str(PCs2plot[0]),'PC'+str(PCs2plot[1])]:
        
        EEMpositions,translations = getEEMpositions2(250,790,260,800,5)
        
        translations = translations.set_index('ids')
        
        rcParams['figure.figsize'] = (1.6, 1.4)
        rcParams['axes.titlesize'] = (7)
        rcParams['axes.titleweight'] = (1000)
        
        # fig,ax = plt.subplots()
        # ax.axhline(y=250, color='grey')
        # ax.axvline(x=260, color='grey')
        # ax.grid(True, which='both')
        EEM4plot = pd.DataFrame(loadings[pc]).T
        
        newpositions = []
        
        # for item in EEM4plot.index:
        #     newpositions.append(translations.loc[item,'coords'])
        
        from aux_functions_2 import plot_partialspectrum
        
        plot_partialspectrum(EEM4plot, EEM4plot, pc,vmin=-0.023,vmax=0.023)
        
        # plotasEEMcontour(newpositions,EEM4plot,[-0.023,-0.017,-0.011,-0.005,0,0.005,0.011,0.017,0.023])
        # plotasEEMcontour(EEMpositions,EEM4plot)
        
        # plt.title("")
        # plt.title(X.index[n])  
        if savefig:          
        
            fig.savefig(path2savefigs+'2DF-loads-'+pc+'_'+data_of_interest+'.png'
                      ,dpi=450,bbox_inches='tight')   
        plt.show()


if data_of_interest == "AbsScan2ext" or data_of_interest == "AbsScan":
    
    
    for pc in ['PC'+str(PCs2plot[0]),'PC'+str(PCs2plot[1])]:
        waves = list(range(300,800))
        newwaves = []
        
    
        for item in EEM4plot.index:
            if data_of_interest == "AbsScan":
                newwaves.append(int(item[7:10]))
            else:
                newwaves.append(int(item[-3:]))
        
        fig,ax = plt.subplots(figsize=(2,1.5))
    
        # plt.plot(waves,EEM4plot*2-2,c="grey",)
            
        EEM4plot2 = loadings[pc].sort_index()
        newwaves = []
        for item in EEM4plot2.index:
            if data_of_interest == "AbsScan":
                newwaves.append(int(item[7:10]))
            else:
                newwaves.append(int(item[-3:]))
        
    
        plt.bar(newwaves,EEM4plot2*2,width=1.5)
        # plt.xlim([400,500])
        # plt.ylim([])
        if savefig:
            fig.savefig(path2savefigs+'Absloads-loads-'+pc+'_'+data_of_interest+'.png'
                      ,dpi=450,bbox_inches='tight')    
        plt.show()    