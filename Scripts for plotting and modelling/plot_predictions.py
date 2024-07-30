# -*- coding: utf-8 -*-
"""
Created on Mon May 13 18:53:41 2024

@author: Pedro

Script for plotting assay-wise predictions and compute statistical analysis

"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

path2savefigs = "C://Users//pedro.brandao//OneDrive - iBET//Python scripts//"+\
                    "Learning PT_version110823//Results//"

model2predict = 'Fx/Noise'
output2predict = "Fx (ppm)"
standard = 'HPLC'
discrimination = 'Tech'
color = 'C4'


fx_predicted = pd.read_excel('fx_predicted.xlsx').set_index('Obs_ID')
sampledata = pd.read_excel('sampledata_Ax.xlsx').set_index('Obs_ID')


#%%

''' Plot all data independent of assay '''

data = sampledata.join(fx_predicted).dropna()

d1 = data.loc[:,['Day','Assay_ori',model2predict]].rename(columns={model2predict:output2predict})
d1['Tech'] = model2predict

d2 = data.loc[:,['Day','Assay_ori',standard]].rename(columns={standard:output2predict})
d2['Tech'] = standard


data2plot = pd.concat([d1,d2])

point = sns.pointplot(x='Day', y=output2predict, hue=discrimination,
            data=data2plot, errorbar='sd' ,markersize = 7.5,linewidth = 1,
            capsize = 0.125, err_kws={'linewidth': 1.5},dodge=0.4)


#%%


figsize = (3.5,3)

''' Plot data dependent of assay '''

discrimination = 'Assay_ori'

data2plot1 = data2plot[data2plot['Tech'].isin([model2predict])].sort_values(by='Assay_ori')

data2plot2 = data2plot[data2plot['Tech'].isin([standard])].sort_values(by='Assay_ori')


plt.figure(figsize=figsize)


point = sns.pointplot(x='Day', y=output2predict, hue=discrimination,
            data=data2plot2, errorbar='sd' ,markersize = 7.5,linewidth = 0,markers=['o','s','^','d'],
            capsize = 0.125, err_kws={'linewidth': 1.5}, 
            palette = ['C7','C7','C7','C7'])

point = sns.pointplot(x='Day', y=output2predict, hue=discrimination,
            data=data2plot1, errorbar='sd' ,markersize = 7.5,marker ='x',
            linewidth = 1,linestyle='--',
            capsize = 0.125, err_kws={'linewidth': 1.5},
            palette = [color,color,color,color])


plt.legend().set_visible(False)
# plt.ylim([0,35])

fig = plt.gcf()

badchars = ["(",")","/"," ","."]
output_name = output2predict
model_name = model2predict

for i in badchars:
    output_name = output_name.replace(i,"")
    model_name = model_name.replace(i,"")
    


fig.savefig(path2savefigs+output_name+'_'+model_name+'_points.png'
              ,dpi=450,bbox_inches='tight')


#%%

models2plot = ['Fx/A-12','Fx/F-2','Fx/BM']
# models2plot = ['CC/A-16','CC/F-3','CC/BM']
standard = 'HPLC'
# standard = 'Muse'
output2predict = "Fx (ppm)"
# output2predict = "CC (M/mL)"

''' Concat all data independent of assay '''

data2plot = pd.DataFrame()

for model in models2plot:

    d = data.loc[:,['Day','Assay_ori',model]].rename(columns={model:output2predict})
    d['Tech'] = model
    
    d[standard] = data[standard]

    data2plot = pd.concat([data2plot,d])


d = data.loc[:,['Day','Assay_ori',standard]].rename(columns={standard:output2predict})
d['Tech'] = standard

d[standard] = data[standard]

data2plot = pd.concat([data2plot,d])

#%%

import colorsys
from matplotlib.ticker import StrMethodFormatter
from emergency_aux_functions import lighten_color
import numpy as np
from scipy.stats import levene,shapiro,kruskal,f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scikit_posthocs as sp


colors = ['C0','C4','C2','C7']

markers = ['x','x','x','o']

legends = True

lightened_colors1 = [lighten_color(c, amount=0.75) for c in colors]

lightened_colors2 = [lighten_color(c, amount=0.5) for c in colors]
                  

for assay in ['F2','F2+N','F2+N+P','F2+N+P_part2']:
    
    y_aux = data2plot[data2plot['Assay_ori'].isin([assay])]
    
    y_aux = y_aux[y_aux['Tech'].isin(models2plot)]
    # y_aux = y_aux.sort_values(by="Tech",ascending = True)
    
    figsize = (2.6,2.3)
    figsize = (4.5,3.75)
    
    plt.figure(figsize=figsize)

    sns.scatterplot(data = y_aux, x = standard, style_order = markers, hue='Tech',
                    y = output2predict, palette = lightened_colors1,
                    legend = legends)
    
    stdy = np.std(y_aux[standard])
    yplotmax = max(np.max(y_aux[output2predict]),np.max(y_aux[standard]))*1.1
    yplotmin = min(np.min(y_aux[output2predict]),np.min(y_aux[standard]))*0.75
    
    
    
    plt.plot([0,yplotmax],[0,yplotmax],'--',color='k',linewidth=0.75) # Y = PredY line
    plt.plot([0,yplotmax],[stdy,yplotmax+stdy],'--',color='k',linewidth=0.75)
    plt.plot([0,yplotmax],[-stdy,yplotmax-stdy],'--',color='k',linewidth=0.75)
    
    plt.xlim([yplotmin,yplotmax])
    plt.ylim([yplotmin,yplotmax]) 
    
    plt.ylabel(output2predict+' - Model')
    plt.xlabel(output2predict+' - '+standard)
    
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
    plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
    plt.tight_layout()
    
    fig = plt.gcf()
                        
    fig.savefig(path2savefigs+"benchmarking_"+output_name+assay,dpi=300)   
    
    plt.figure(figsize=figsize)
    # plt.xlim([yplotmin,yplotmax])
    plt.ylim([yplotmin,yplotmax]) 
    
    y_aux = data2plot[data2plot['Assay_ori'].isin([assay])]
    
    
    sns.swarmplot(x='Day', y=output2predict, hue='Tech', data=y_aux,
                  dodge=0.2, palette = lightened_colors2,zorder=1,legend = False)
    
    sns.pointplot(x='Day', y=output2predict, hue='Tech', data=y_aux,
                  dodge=0.6, capsize=0.1, err_kws={'linewidth': 1},
                  markersize = 7,
                  linewidth = 0.5, palette =colors,zorder=100,legend = legends)
    
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
    plt.tight_layout()
    
    fig = plt.gcf()
                        
    fig.savefig(path2savefigs+"benchmarking_"+output_name+assay+"assay-wise",dpi=300)    



shapiro_results = pd.DataFrame(index=models2plot+['HPLC'])
levenes_results = pd.DataFrame()
anova_results = pd.DataFrame()
posthoc_results = pd.DataFrame()

for assay in ['F2','F2+N','F2+N+P','F2+N+P_part2']:
    
    levenes = pd.DataFrame()
    anovas = pd.DataFrame()
    posthocs = pd.DataFrame()
    shapiros = pd.DataFrame()
    
    print(assay)
   
    # Filter the data for the current assay
    data2plot_aux = data2plot[data2plot['Assay_ori'] == assay]
    
    # Perform Shapiro-Wilk test to check for normality
    for name, group in data2plot_aux.groupby(['Tech']):
        
        shapiro_test_stat, shapiro_p_value = shapiro(group[output2predict].values)
        print(name, np.round(shapiro_p_value, 3))
        
        # Properly indexing the DataFrame to store the results
        shapiros.loc[name[0],assay] = np.round(shapiro_p_value, 3)
        
    # Group by 'Tech' and gather the data for Levene's test and ANOVA
    groups = [group[output2predict].values for name, group in data2plot_aux.groupby('Tech')]
    
    # Perform Levene's test
    levene_stat, levene_p_value = levene(*groups)
    print(f"Levene's test p-value: {np.round(levene_p_value, 3)}")
    
    # Store Levene's test result
    levenes.loc[assay,'Levene_p_value'] = np.round(levene_p_value, 3)
    
    if levene_p_value >= 0.05 and shapiros.min().iloc[0] >= 0.05:  # If Levene's test is not significant, perform ANOVA
        anova_stat, anova_p_value = f_oneway(*groups)
        print(f"ANOVA p-value: {np.round(anova_p_value, 3)}")
        
        # Store ANOVA result
        anovas.loc[assay,'p-value'] = np.round(anova_p_value, 3)
        anovas.loc[assay,'test'] = "ANOVA"
        
        if anova_p_value < 0.05:  # If ANOVA is significant, perform Tukey's HSD test
            tukey = pairwise_tukeyhsd(endog=data2plot_aux[output2predict], groups=data2plot_aux['Tech'], alpha=0.05)
            posthocs = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])
            posthocs = posthocs[['group1','group2','p-adj']]
            posthocs['Assay'] = assay
    else:
        anova_stat, anova_p_value = kruskal(*groups)
        print(f"Kruskall p-value: {np.round(anova_p_value, 3)}")
        anovas.loc[assay,'p-value'] = np.round(anova_p_value, 3)
        anovas.loc[assay,'test'] = "Kruskal"
        
        if anova_p_value < 0.05:  # If ANOVA is significant, perform Tukey's HSD test
            posthocs = sp.posthoc_dunn(data2plot_aux, val_col=output2predict, group_col='Tech', p_adjust='bonferroni')
            mask = np.triu(np.ones((len(posthocs), len(posthocs))), k=1).astype(bool)
            posthocs = posthocs.stack()[mask.ravel()].reset_index()
            posthocs['Assay'] = assay

    
    print(posthocs)
    
    shapiro_results = shapiro_results.join(shapiros)
    levenes_results = pd.concat([levenes_results,levenes])
    anova_results = pd.concat([anova_results,anovas])
    posthoc_results = pd.concat([posthoc_results,posthocs])
    

