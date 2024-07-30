**Spectroscopy-based Machine Learning models for monitoring of Fucoxanthin Productivity and Cell Growth in Phaeodactylum tricornutum cultures**

![PT-fucoFromSpec](https://github.com/user-attachments/assets/9aac9016-9952-4cc3-be15-b0f9e0e818e3)

This repository includes the data set generated from the monitoring of P. tricornutum cultivations through Cytometry, HPLC, 2D-fluorescence and Absorbance. There are also two folders containing the scripts.

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
