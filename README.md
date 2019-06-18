# pyclust
This is a set of scripts, and results, that compares the different clustering options in python's sklearn, to the mclust package in R

In the main directory we have a few items:

brute_cluster_experiments.py performs "brute clustering" on the datasets in the data folder - START BY RUNNING THIS SCRIPT \
brute_cluster.py - contains the functions needed to perform brute clustering \
brute_cluster.r - uses mclust to search over clustering methods to find that best BIC and ARI \
	Note: technically, I let mclust calculate BIC its own way, but I have verified its calculations on several examples \
data directory - various datasets \
figures directory - various figures


In the compare_bic directory, we directly compare the analagous methods in mclust and python:

r_create_hc.r - perform different options of hierarchical agglomeration and save the results \
python_create_hc.py - analog to above file \
pyton_em.py - reads in the results of hierarchical agglomerations then performs EM then saves BIC \
r_em.r -reads in the results of hierarchical agglomerations then performs EM then saves the parameters in csvs in the folder r_em_params \
calc_bic_r.py - reads the parameters in the r_em_params directory then calculates and saves bic \
bic_plots.opy - read the bic csv results and compares graphs the results, comparing python and R head to head \
bic.py - contains functions used to calculate BIC \
Various images made by bic_plots.py \
compare_bic - latex document that presents the images

