Spectroscopy-based Machine Learning models for monitoring of Fucoxanthin Productivity and Cell Growth in Phaeodactylum tricornutum cultures

This repository includes the data set generated from the monitoring of P. tricornutum cultivations through Cytometry, HPLC, 2D-fluorescence and Absorbance.

The data set generated in this work is organized into 6 dataframes (.pkl).

These include for all observations (i.e., culture samples) fluorescence and absorbance data, standard analytical data, and information data (i.e., date, assay type, etc.).

Description for the data set excel worksheet files
File	Description
sampledata.pkl	Labels for each observation within the data set, associating it with the assay and batch of origin, and the date collected; includes alternative Ids for the assays, for the batches, and date/day collected
monitordata.pkl	Contains the data measured by standard analytical methods for each observation
fluorodata.pkl	Contains the data measured by 2D-fluorescence for each observation
absdata.pkl	Contains the data measured by absorbance spectroscopy for each observation
absextdata.pkl	Contains the data measured by absorbance spectroscopy of the methanol extract of the samples for each observation
no3po4data.pkl	Contains NO3 and PO4 measurements obtained from the sample supernatants for each observation

Several python scripts were developed in-house with purposes of data importation, handling and plotting, statistical and multivariate analysis, and machine learning regression.

Description for the python script files developed in-house
File	Description
scrip_ML_PTax.py	Script for loading the data set and applying ML algorithms together with spectral variable selection methods and data pre-processing transformations
allspectralelement_linearReg.py	Script for loading the data set and applying single spectral element linear regression to all elements of the spectrum
datasetplotter_script.py	Script for plotting the data set, includes cells for plotting fluorescence spectra, absorbance spectra, and standard analytical data. Also, statistical analyis pipeline is included
PCA_script.py	Script for loading the data set and performing PCA analysis, including score and loading plots
stats_auxfunctions.py	Script with functions for applying statistical analysis pipeline
plot_predictions.py	Script for plotting assay-wise predictions and compute statistical analysis
aux_functions.py	Auxiliary functions for data augmentation, fitting ML model (CNN, PLS, XGB, RFR or SVM), running PLS models with windows within the spectra, VIP, dataset split based on labels, plotting accuracy scatters, dilution factor application by day/assay, data transformations, linear calibration
aux_functions_2.py	Auxiliary functions for: collecting the data from excel files, building CNN architectures, computing accuracy, variable selection within spectrum, coordenate generation for spectrum selection, convert 2D-EEM to 1D and vice-versa, for applying CNNR_2D, plotting results, color code conversions

