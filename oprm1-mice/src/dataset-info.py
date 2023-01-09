#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 14:50:22 2022

@author: konstantinos kalaitzidis
"""

# print information for the data files


def info(datafile_names):

    def names(int):
        names = ["Behavioral", "Tracking", "h5"]
        return print(names[i], "INFORMATION AND MISSING VALUES:")

    i = 0
    for key in datafile_names:
        names(i)
        print(datafile_names[key].info(verbose=False), "\n")
        print(datafile_names[key].isnull().sum(), "\n")
        i = i + 1
