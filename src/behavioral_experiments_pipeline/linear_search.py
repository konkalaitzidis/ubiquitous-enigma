#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 16:46:23 2022

@author: konstantinos kalaitzidis
"""


def l_search(values, search_for):

    search_at = 0
    search_res = False
    count = 0
    detect_time = 0
    values = values.iloc[:, 7]

    # Match the value with each data element
    while search_at < len(values, 1) and search_res is False:
        if values[search_at] == search_for:
            search_res = True
            detect_time = values.iloc[count-1, 0]
            print("Index Row is:", count,
                  "and the time of ca detection is:", detect_time)
            return detect_time
        else:
            search_at = search_at + 1
            count = count+1


# #see where we have the firt ca occurnace and print it's timestamp
# def l_search(values, search_for):

#     search_at = 0
#     search_res = False
#     count = 0
#     detect_time = 0

#     #Match the value with each data element
#     while search_at < len(values) and search_res is False:
#       if values[search_at] == search_for:
#           search_res = True
#           detect_time = beh_data.iloc[count-1, 0]
#           print("Index Row is:", count, "and the time of ca detection is:", detect_time)
#           return detect_time
#       else:
#           search_at = search_at + 1
#           count = count+1

# calcium_detection_times = beh_data.drop_duplicates(subset=['0.3'])

# # see where we have the first ca occurnace and print it's timestamp
# detect_time = l_search(beh_data.iloc[:, 7], 1)
