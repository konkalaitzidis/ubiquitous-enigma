#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 14:50:22 2022

@author: konstantinos kalaitzidis
"""
from codeAdaptation import datafile_names, beh_data, dlc_data, f

#print information for the data files
def dataset_info():
    
        def data_names(int):
            names = ["Behavioral", "Tracking", "h5"]
            return print(names[i], "INFORMATION:")
        
    
        datafile_names = {'behavioral data' : beh_data, 'tracking data': dlc_data, 'h5':f}
        i = 0
        for key in datafile_names:
            data_names(i)
            print(datafile_names[key].info(), "\n")
            i = i + 1