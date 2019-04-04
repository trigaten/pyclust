# pyclust
This is a set of scripts, and results, that compares the different clustering options in python's sklearn, to the mclust package in R

In the compare_bic directory, we directly compare the analagous methods in mclust and python:

r_create_hc.r - perform different options of hierarchical agglomeration and save the results
python_create_hc.py - analog to above file
pyton_em.py - reads in the results of hierarchical agglomerations then performs EM then saves BIC
r_em.r -reads in the results of hierarchical agglomerations then performs EM then saves the parameters in csvs in the folder r_em_params
calc_bic_r.py - reads the parameters in the r_em_params directory then calculates and saves bic
bic.py - contains functions used to calculate BIC
	