# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 17:00:57 2021

@author: Pedro

Script for plotting spectrum and culture dynamics

Also, it can perform statistical analysis

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.interpolate import interp2d
from matplotlib import rcParams
from aux_functions_2 import collectdata

import seaborn as sns

from aux_functions import augment_df_byday,linearcalibration

folder = "C:/Users/pedro.brandao/OneDrive - iBET/Documents - PB/Fuco2D_May24/Suppplementary Attachments/S1/DataSet"
folder4augmenteddata = "C:/Users/pedro.brandao/OneDrive - iBET/Documents - PB/Fuco2D_May24/Suppplementary Attachments/S2.3"


# Collect the data

monitordata,sampledata,fluorodata,absdata,absextdata,no3po4data = collectdata(folder)

monitordata_ori = monitordata.copy()
fluorodata_ori = fluorodata.copy()

badchars = ["(",")","/"," ","."]
SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
Trainsub = str.maketrans("t","ₜ")

monitordata = monitordata_ori.copy()

#%%

"""
This cell serves for performing linear calibrations between variables
and creates new variables, such as Fx (mg/g) or Chl per cell 

"""

monitordata = monitordata_ori.copy()


# Obtain parameters from linear calibrations with standard analytical equipment

monitordata['MUSE DW'] = 1e-3*monitordata['CC (M/mL)'] * np.log(monitordata['CytoFSC (a.u.)']).pow(3)

fit_intercept = True
logtransform = False
samples2consider = sampledata # [sampledata['Assay_ori'].isin(['F2','F2+N','F2+N+P','F2+N+P_part2','Fed F2'])]

samples2consider = sampledata[sampledata['Assay_ID'].isin(['PTop','PTT1','PTT2'])]

standard = monitordata['DW - experimental (g/L)'].dropna()


wave = pd.Series(monitordata['MUSE DW'])

# standard = absextdata['ext_445']
# wave = absdata['Abs at 445 nm']

m0,b0 = linearcalibration(samples2consider,standard,wave,fit_intercept,logtransform)

                
monitordata['Fx (pg/cell)'] = monitordata['Fx (ppm)']/monitordata['CC (M/mL)']
monitordata['DW (g/L)'] = m0*monitordata['MUSE DW'] + b0


#%%

df = pd.DataFrame()
df = sampledata.join(monitordata)
df = df.join(no3po4data)
df = df.join(absextdata)
df = df.join(fluorodata)
df = df.join(absdata)


criterion = 'Assay_ori'
data2use = ['F2','F2+N','F2+N+P','F2+N+P_part2']



discriminator = 'Assay_ori'
colors = ['C0','C1','C2','C3']

colors = [(1.0, 0.8352941176470589, 0.4745098039215686),
          (0.7490196078431373, 0.5647058823529412, 0),
          (0.4980392156862745, 0.3764705882352941, 0),
          (0.34901960784313724, 0.2627450980392157, 0),
          'C4']

symbols = ['o','s','^','x','d','^','s','d','x']

varyID = list(monitordata.columns)


viability_tres = 0

metabatches2remove = ['PTS1','PTS2','PTax3']

batches2remove = ['PTT1_01','PTT1_05','PTT1_07','PTT1_18','PTT1_20','PTT1_21','PTT1_22',
                  'PTop_07','PTop_02','PTS14',]

# batches2remove = ['PTT1_01','PTT1_04','PTT1_05','PTT1_10','PTT1_13','PTT1_18','PTT1_19']

assays2remove = [''] 



obs2remove =  []

days2remove = [] # ['Friday, September 30, 2022']


obs2remove_aux = list(df[df['Viability (%)'] < viability_tres].index)

if len(obs2remove_aux) > 0:
    obs2remove += obs2remove_aux

vars2remove = ['Viability (%)']


# df = df.drop(index = 'PTop_17_4')

# Bar plots for comparing 2-way ANOVA time and assay
for var in vars2remove:
    varyID.remove(var)

df_work = df.copy()    

# Remove observations 

df_work = df_work[[not elem for elem in df_work.index.isin(obs2remove)]]
df_work = df_work[[not elem for elem in df_work.Batch_ID.isin(batches2remove)]]
df_work = df_work[[not elem for elem in df_work.Assay_ID.isin(metabatches2remove)]]
df_work = df_work[[not elem for elem in df_work.Assay_ori.isin(assays2remove)]]

# df_work = df_work[[not elem for elem in df_work.Date.isin(days2remove)]]

# df_work = df_work[df_work['Batch_ID'].isin(batches2remove)].dropna()

df_work = df_work[df_work[criterion].isin(data2use)]
discriminators = df_work[criterion].unique()

# df_work = df_work[df_work[discriminator].isin(discriminators)]

assay_ID = df['Assay_ID'].to_list()
assays = list(dict.fromkeys(assay_ID))




#%%

# Plot EEMs

path2savefigs = "C://Users//pedro.brandao//OneDrive - iBET//Imagens//micbiotec23//"

rcParams['figure.figsize'] = (1.8,1.8)
rcParams['axes.titlesize'] = (7)
rcParams['axes.titleweight'] = (1000)

iterator = 0

assays2use = ['F2','F2+N','F2+N+P']

discriminator = 'Assay_ori'
factors = assays2use

plotdiffs = False
logtransf = True

days2use = [3,5,7,9]

varxID = list(fluorodata.columns)


for assay in factors:

    df4plot = df_work[df_work[discriminator].isin([assay])].copy()
    
    # days2use = list(dict.fromkeys(list(df4plot['Day'])))
    # days2use.sort()
    
    iteration = 0
    
    for day in days2use:
        iteration += 1
        df4plot_aux = df4plot[df4plot['Day'].isin([day])]
        
        if len(df4plot_aux) != 0:
            
            iterator +=1

            X = df4plot_aux[varxID].copy()
            
            if logtransf:
                X[X<1e-3] = 1e-3
                X = np.log(X)            
            
            Y = df4plot_aux[varyID].copy()  
            
            x = X.mean().T

            if iteration == 1:
                x_aux = X.mean().T
            
            if plotdiffs:
                
                x -= x_aux
                # x = np.abs(x)
                x_aux = X.mean().T
            
            # for n in range(len(X)):
                
            #     x = X.iloc[n]
                    
            if not x.isna().any():
           
                z = np.zeros((109,109)) # matrix with 0's
                positions = [] # EEP coordinates in the EEM
                
                ex = np.arange(250,795,5)
                em = np.arange(260,805,5)
                
                iteration = 0
                
                
                for i in range(109):
                    for j in range(109):
                   
                        if ex[i] >= em[j]:
                            z[i,j] = 0
                        else:
                            z[i,j] =x[iteration]

                            positions.append([i,j]);
                            iteration +=1
                    
                        
                xplot = em
                yplot = ex
                zplot = z
            
                f = interp2d(xplot, yplot, z, kind='linear')
                ynew = np.arange(250,790.5,0.5)
                xnew = np.arange(260,800.5,0.5)
                data1 = f(xnew,ynew)
                

                
                # Define the colors for the colormap in custom mode
                colors = [(0.0, 'cyan'),(0.2,'dodgerblue'),
                          (0.4, 'black'), (0.6, 'black'), (0.8, 'red'),
                          (1.0, 'yellow')]
                
           
                
                contourf = plt.contourf(xnew, ynew, data1, #[-1,-0.5,-0.3,-0.1,0,0.1,0.3,0.5,1],
                                        cmap='turbo') #,vmax=0.5,vmin=-0.5)

                plt.contour(xnew, ynew, data1, levels=contourf.levels, colors='white')
                
                # plt.colorbar()

                figID = '2DF of T'+str(day)+' of '+assay
                plt.title(figID)
                # plt.title(X.index[n])            
                # plt.colorbar()
                # plt.savefig(str(iterator)+'_Average 2DF of T'+str(day)+' of assay '+assay,dpi=300)
                fig=plt.gcf()
                # fig.savefig(path2savefigs+str(iterator)+'_Average 2DF of T'+str(day)+' of assay '+assay+'.png'
                #               ,dpi=300,bbox_inches='tight') 
                plt.show()


#%%

rcParams['figure.figsize'] = (3.25, 2.5)
rcParams['figure.figsize'] = (12.5, 10)
rcParams['axes.titlesize'] = (7)
rcParams['axes.titleweight'] = (1000)

# Plot spectrum by spectrum

spectra = 'AbsExt'


discriminator = 'Assay_ori'

assays = ['F2','F2+N']

days2use = [3,5,7,9]

plotdiffs = False
logtransf = False

ex = np.arange(300,800,1)

if spectra == 'AbsScan':
    abs2plot = list(absdata.columns)

    
elif spectra == "AbsExt":
    abs2plot = list(absextdata.columns)
    


varxID = abs2plot
iterator = 0

df = df_work.copy()

for assay in assays:
    
    df4plot = df[df[discriminator].isin([assay])].copy()
    iteration = 0

    for day in days2use:
        
        iteration += 1
        df4plot_aux = df4plot[df4plot['Day'].isin([day])]
        
        if len(df4plot_aux) != 0:
            
            iterator +=1
    
            X = df4plot_aux[varxID].copy()
            
            if logtransf:
                X += 1
                X = np.log(X)   
            
            # for n in [6]:
                
            #     x = X.iloc[n] 
            
            x = X.mean()
            
            if iteration <= 1 and plotdiffs:
                x_aux = X.mean()
            
            if plotdiffs: 
                x -= x_aux
                # x = np.abs(x)
                x_aux = X.mean()
    
            # for n in range(len(X))[4:9]:  
            #     x = X.iloc[n]
            # sns.set_palette([(1.0, 0.8352941176470589, 0.4745098039215686),
            #                  (0.7490196078431373, 0.5647058823529412, 0),
            #                  (0.4980392156862745, 0.3764705882352941, 0),
            #                  (0.34901960784313724, 0.2627450980392157, 0)])
            if plotdiffs:
                if iteration > 1:
                    plt.plot(ex,x,label= x.name)#'DAI '+str(day))
            else:
                plt.plot(ex,x,label= x.name)#'DAI '+str(day))
             
            # plt.xlim([5,7])
            
        plt.legend(days2use)
        
        plt.title('Abs Scans of '+assay)
        
        plt.ylim([-0.001,1.75])
        
        
        fig=plt.gcf()
        # fig.savefig(path2savefigs+'abs'+assay+'_300dpi.png'
        #                 ,dpi=300,bbox_inches='tight') 
    plt.show()  





#%%


from stats_auxfunctions import run_stats
from aux_functions_2 import mpl_color_to_rgb

path2savefigs = "C://Users//pedro.brandao//OneDrive - iBET//Imagens//PTfuco2D//"
# path2savefigs = 'C://Users//Pedro//OneDrive - iBET//Imagens//PTfuco2D//'

# figsize = (3,2.8)
figsize = (1.25,1.2)
# figsize = (7,6)

simulate_dataaugment = 0
plotnumblim = 1
shift = 0
augnum = 1
days2use = [3,5,7,9,12,14,16,18,20]

stats = False

colors = [(0.9, 0.7352941176470589, 0.3745098039215686),
           (0.7490196078431373, 0.5647058823529412, 0),
          
          (0.4980392156862745, 0.3764705882352941, 0),
           (0.3980392156862745, 0.2764705882352941, 0)]

# colors = [
#            (0.3980392156862745, 0.2764705882352941, 0)]

# colors = [(0.3980392156862745, 0.2764705882352941, 0),
# (1.0, 0.8352941176470589, 0.4745098039215686)]
          
# colors = [(0.4980392156862745, 0.3764705882352941, 0)]



# colors = [(1.0, 0.8352941176470589, 0.4745098039215686),
#           (0.7490196078431373, 0.5647058823529412, 0),
#           (0.3980392156862745, 0.2764705882352941*0.5, 0.6),
#           (0.4980392156862745, 0.3764705882352941, 0),
#           (0.4980392156862745, 0.3764705882352941, 0),
#           (1.0, 0.8352941176470589, 0.4745098039215686),
#           ]

# colors_user = ['C2','C0','C1','C3']

# colors = list(mpl_color_to_rgb(val) for val in colors_user)


symbols = ['o','s','^','d','d','^','s','d','x']

# symbols = ['d']

# symbols = ['s','o','x','^','d','D']

discrimination = 'Assay_ori'

discriminators = df_work[discrimination].unique()

extraT0value = 0.01

numofgraphs = 0

while numofgraphs < plotnumblim:
    
    numofgraphs += 1
    
    for output2predict in ['PO4 (uM)']:
        
        if days2use == 'all':
            days2use = list(dict.fromkeys(list(df_work['Day'])))
            days2use.sort()
        
        
        df4plot = df_work[['Day','Assay_ori','Assay','Assay_ID',output2predict]]
        df4plot = df4plot.sort_values(by='Assay_ori',ascending=True)
        days = list(dict.fromkeys(list(df4plot['Day'])))
        days.sort()
        df4plot = df4plot[df4plot['Day'].isin(days2use)].copy()
        
        
        if simulate_dataaugment:
                 
            df4plot = augment_df_byday(df4plot,augnum,shift,output2predict)

        y4plot = df4plot[[output2predict,'Day','Assay_ori']].copy()
        
        y4plot[output2predict] = pd.to_numeric(y4plot[output2predict],errors='coerce')
        
        print('\n=================== '+output2predict+' =====================\n',)
        
   
        
        if stats:
            
            testused,p_value,posthoc_results = run_stats(y4plot,output2predict,'Assay_ori')
            print(posthoc_results,'\n')
            
            for discriminator in ['F2+N']: #discriminators:
                
                print(discriminator)
                
                data = y4plot[y4plot[discrimination] == discriminator].reset_index().dropna()
                factor = 'Day'
                testused,p_value,posthoc_results = run_stats(data,output2predict,factor)
                
                print(posthoc_results,'\n')
                             
            testused,p_value,posthoc_results = run_stats(y4plot,output2predict,'Day')
            print(posthoc_results,'\n')        
            
            for day in [3,5,7,9]:
                
                print(day)
            
                data = y4plot[y4plot['Day'] == day].reset_index()
                factor = 'Assay_ori'
                testused,p_value,posthoc_results = run_stats(data,output2predict,factor)
                
                print(posthoc_results,'\n')
        
        output_name = output2predict
        
        for i in badchars:
            output_name = output_name.replace(i,"")
            
        df4plot_aux = df4plot[df4plot['Day'].isin(days2use)].copy()          
        df4plot_aux = df4plot_aux[df4plot_aux[discrimination].isin(discriminators)].dropna()
            
        df4plot = df4plot.rename(columns={'output':output2predict})
        
        # Plot means and standard deviations
        
        plt.figure(figsize=figsize)
  
        lighten_factor = 0.25
        
        lightened_colors = []

        for color in colors:
            lightened_color = tuple(min(1.0, channel + lighten_factor) for channel in color)
            lightened_colors.append(lightened_color)
        
        # lightened_colors = colors
        
        # Replace 'NaN' with actual values if 'output2predict' and 'Assay_ori' must have data
        # new_rows = {'Day': 10, 'output2predict': pd.NA, 'Assay_ori': pd.NA}
        
        # Append the new row
        # df4plot_aux = df4plot_aux.append(new_rows, ignore_index=True)
       
        swarm = sns.swarmplot(x='Day',y=output2predict,hue=discrimination,
                    data=df4plot_aux,dodge=0,
                    palette = lightened_colors,size=3,zorder=1,legend = False)
          
        
        point = sns.pointplot(x='Day', y=output2predict,hue=discrimination,
                    data=df4plot_aux, errorbar='sd' ,markersize = 7.5,linewidth = 1,
                    capsize = 0.125, err_kws={'linewidth': 1.5},dodge=0,
                    markers = symbols,palette = colors,zorder=100) #, linestyles = '')
       

        plt.legend().set_visible(False)
        
        
        # plt.yticks([0,4,8,12,16])
        plt.ylim([-5,30])
        plt.xlim([-.25,4.5])
            
        fig = plt.gcf()
        fig.savefig(path2savefigs+output_name+'_'+data2use[0]+'_'+\
                    'dataaug_'+str(simulate_dataaugment)+'_points.png'
                      ,dpi=450,bbox_inches='tight')

        
        fig = plt.gcf()
        fig.savefig(path2savefigs+output_name+str(numofgraphs)+'_swarm.png'
                      ,dpi=300,bbox_inches='tight')   
        plt.show()       



