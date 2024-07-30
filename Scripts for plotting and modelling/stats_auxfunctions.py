# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 12:19:11 2024

@author: pedro.brandao

Scripts for applying statistical analysis pipeline

"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import levene,shapiro,kruskal
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scikit_posthocs as sp


def run_stats(y4plot,output2predict,factor):
    
    # Run Shapiro-Wilk test: H0 -> dataset is normally distributed
    
    minimal_shapiro_pvalue = 1
    problematic_assays = []
    
    for name, group in y4plot.groupby(factor):
        shapiro_test_stat, shapiro_p_value = shapiro(group[output2predict].values)
        print(str(name)+' Shapiro-Wilk p-value = ',np.round(shapiro_p_value,3))
        if shapiro_p_value < minimal_shapiro_pvalue:
            minimal_shapiro_pvalue = shapiro_p_value
            problematic_assays.append((name,shapiro_p_value))
    
    # Run Levene test: H0 -> dataset is homoscedastic
    
    levene_stat, levene_p = levene(*[group[output2predict].values for name, group in y4plot.groupby(factor)])
    print(f"Levene's Test for {factor} effect: p-value = {levene_p:.3f}")
    
    # Run 2-way statistics
    
    if levene_p < 0.05 and minimal_shapiro_pvalue >= 0.01 :
        
        # Run Welch's ANOVA 
        print("Warning: Variances are not equal (p < 0.05 in Levene's test). Using Welch's ANOVA.")
        testused = "Welch's"
    
        formula = f'Q(output2predict) ~ C({factor})'
        model = ols(formula, data=y4plot).fit()
        anova_table = sm.stats.anova_lm(model, typ=3, robust='hc3')  # Using heteroscedasticity-consistent covariance matrix estimator
        p_value = anova_table['PR(>F)'].iloc[1]
        print('Welchs ANOVA p_value = ',p_value)
        # Tukey's HSD pairwise test 
        tukey = pairwise_tukeyhsd(endog=y4plot[output2predict], groups=y4plot[factor], alpha=0.05)
        posthoc_results = pd.DataFrame(data=tukey._results_table.data[1:],
                                       columns=tukey._results_table.data[0]).set_index('group2')
    
                  
    elif minimal_shapiro_pvalue <0.01:
        # Run Krukal-Wallis
        print(f"Warning: Low likelyhood of normality of at least one of the {factor}" +\
              "(p < 0.05 in Shapiro-Wilk's test). Using Kruskal-Wallis.")
        testused = "Kruskal-Wallis"
        # Perform Kruskal-Wallis Test
        kruskal_test_stat, p_value = kruskal(*[group[output2predict].values for name, group in y4plot.groupby(factor)])
        
        print(f"Kruskal-Wallis p-value = {p_value:.3f}")
        
        # Dunn's test
        posthoc_results = sp.posthoc_dunn(y4plot, val_col=output2predict, group_col=factor) #, p_adjust='bonferroni')
                  
    else:
        # Run standard ANOVA
        testused = "1-way ANOVA"
        formula = f'Q(output2predict) ~ C({factor})'
        model = ols(formula, data=y4plot).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        p_value = anova_table['PR(>F)'].iloc[0]
    

        print('ANOVA p_value = ',p_value)      
        
        # Tukey's HSD pairwise test 
        tukey = pairwise_tukeyhsd(endog=y4plot[output2predict], groups=y4plot[factor], alpha=0.05)
        posthoc_results = pd.DataFrame(data=tukey._results_table.data[1:],
                                       columns=tukey._results_table.data[0]).set_index('group2')

        
    return testused,p_value,posthoc_results
