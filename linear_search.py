#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 14:51:26 2022

@author: konstantinos kalaitzidis
"""
from main import beh_data

#see where we have the firt ca occurnace and print it's timestamp
def search(values, search_for):
    
    search_at = 0
    search_res = False
    count = 0
    detect_time = 0
    
    #Match the value with each data element	
    while search_at < len(values) and search_res is False:
      if values[search_at] == search_for:
          search_res = True
          detect_time = beh_data.iloc[count-1, 0]
          print("Index Row is:", count, "and the time of ca detection is:", detect_time)
          return detect_time
      else:
          search_at = search_at + 1
          count = count+1