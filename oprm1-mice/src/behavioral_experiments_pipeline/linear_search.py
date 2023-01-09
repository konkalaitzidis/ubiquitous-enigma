#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 16:46:23 2022

@author: konstantinos kalaitzidis
"""


def l_search(values, search_for, t):

    search_at = 0
    search_res = False
    count = 0

    # Match the value with each data element
    while search_at < len(values) and search_res is False:
        if values[search_at] == search_for:
            search_res = True
            t = t.iloc[count-1]
            print("Calcium detection starts at index Row is:", count, " and at time t=", t)
            return t
        else:
            search_at = search_at + 1
            count = count+1
