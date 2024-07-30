In this work, data augmentation was used to simulate technical replication. Data augmentation is a statistical technique popularly used in machine learning, and it provides the ability to simulate an expected form of irrelevant variance in a data set, improving its quality.

In this case, the irrelevant variance to mitigate is human experimental error on the standard analytical methods side. For this purpose, a 5-fold data augmentation was applied, following these steps:

1)	Separately collect the data on the Day of each assay (e.g, Day 3 of assay F/2, Day 5 of assay F/2, …, Day 20 of assay F/2+N+P)
2)	
3)	For each, generate normal distributions for fucoxanthin and cell count, based on their individual mean and standard deviation
4)	
5)	Generate 4 artificial technical replicates for each assay’s Day, by random sampling of the generated distribution
6)	
7)	Create 4 new observations per assay/Day, using the new values of Fx and CC, while just repeating the spectroscopy values
8)	
9)	The data augmentation process was only applied for the training subset.

