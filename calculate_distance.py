#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 14:46:05 2022

@author: pierre.le.merre
"""

#function to calculate distance between the coordinates
def calculate_distance(x1, x2, y1, y2):
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance
