#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 14:51:12 2022

@author: konstantinos kalaitzidis
"""

from codeAdaptation import datafile_names, beh_data, dlc_data, f, sns



#determine if your dataset has missing values 
def missing_info_data():
    print("Behavioral:\n",beh_data.isnull().sum(), "\n")
    print("Tracking:\n",dlc_data.isnull().sum(), "\n")
    print("Calcium:\n",f.isnull().sum(), "\n")
    sns.heatmap(beh_data.isnull())
    sns.heatmap(dlc_data.isnull())
    sns.heatmap(f.isnull())