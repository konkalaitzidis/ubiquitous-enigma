#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 13:15:00 2022

@author: konstantinos kalaitzidis

A module-level docstring

This is the a draft custom-made pipeline to analyze behavioral, tracking, and
calcium imagery data.
"""
# %% Importing Packages and Libraries

# Numeric analysis
from linear_search import l_search
from scipy.stats import gaussian_kde
import pandas as pd
import numpy as np

# Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns


# Functions
import calculate_distance
from dataset_info import info


# %% Data Preparation
print("\n\n\n=====> Data Preparation... <===== \n\n\n")

# Preparing behavioral data file
# path: /Users/pierre.le.merre/OneDrive - KI.SE/Mac/Desktop/KIlab/data files/Behavioral Region of Interest/tmaze_2021-10-16T17_05_25.csv
beh_data_path = input("Insert path of behavioral data file here: ")
beh_data = pd.read_csv(beh_data_path)


# Renaming column '2021-10-16T17:05:26.6032384+02:00' to 'Time'
beh_data = beh_data.rename(columns={
    '2021-10-16T17:05:26.6032384+02:00': 'Time', '0': 'Choice',
    '0.1': 'Init-Reward', 'False': 'Initiation',
    'False.1': 'Incorrect', 'False.2': 'Reward'})


# Preparing deep lab cut file
# /Users/pierre.le.merre/OneDrive - KI.SE/Mac/Desktop/KIlab/data files/Deep lab cut/tmaze_2021-10-16T16_39_15DLC_resnet50_Oprm1_insc_VAS_reversalNov18shuffle1_1030000.h5
dlc_data_path = input("Insert path of dlc data file here: ")
dlc_data = pd.read_hdf(dlc_data_path)


# Add a column named "Time"
dlc_data[dlc_data.shape[1]] = 0
dlc_data = dlc_data.rename(columns={24: 'Time'})


# Reading the h5 file that contains the deeplabcut and calcium imaging
# path: /Users/pierre.le.merre/OneDrive - KI.SE/Mac/Desktop/arrowmaze_project-main/arrowmaze_data2.h5
pathname = input("Insert path of h5 file here: ")
#pathname = "" + str(pathname) + ""
#pathname = "/Users/pierre.le.merre/OneDrive - KI.SE/Mac/Desktop/arrowmaze_project-main/striatum-2choice/data/arrowmaze_data2.h5"
with pd.HDFStore(pathname) as hdf:
    # This prints a list of all group names:
    print("Reading the h5 file that contains the deeplabcut and calcium imaging data...")

h5_file = pd.read_hdf(pathname, key="/meta")


# Removing the date part from the datetime values to keep only seconds
beh_data['Time'] = pd.to_datetime(beh_data['Time'])
beh_data.iloc[:, 0] = beh_data.iloc[:, 0] - beh_data.iloc[0, 0]
beh_data['Time'] = beh_data.iloc[:, 0].dt.total_seconds()


# Adding the time to the dlc_data file with the according step.
'''
Finding the step:
    total time (last time element) /
    the number of row of the dlc file
'''
total_time = beh_data.iat[len(beh_data) - 1, 0]
step = total_time / len(dlc_data)

# Fill in data in the last column (Time column) with the time
dlc_data.iloc[:, 24] = range(len(dlc_data))*step
print("Added TIME values to the dlc file with step", step)


# %% Understanding our datasets
print("\n\n\n=====> Understanding our datasets <===== \n\n\n")


datafile_names = {'behavioral data': beh_data,
                  'tracking data': dlc_data, 'h5': h5_file}
info(datafile_names)


# %% Store processed data

# %% Processing
print("\n\n\n=====> Processing... <===== \n\n\n")

'''Calculate the average speed, the standard deviation, and the standard error of the mean
of the mouse on the behavioral file according to the coordinates on the dlc file'''


# List where speed values will be stored
speed_list = []


# Find all the speeds of the mouse
for index, row in dlc_data.iterrows():

    # control if to exit the function
    if dlc_data.iat[index, 24] == dlc_data.iat[-1, 24]:
        print("All speed values have been stored in list successfully.")
        break

    x1 = dlc_data.iat[index, 18]
    x2 = dlc_data.iat[index+1, 18]
    y1 = dlc_data.iat[index, 19]
    y2 = dlc_data.iat[index + 1, 19]

    distance = calculate_distance.dist_calc(x1, x2, y1, y2)
    speed = distance / (dlc_data.iat[index+1, 24]-dlc_data.iat[index, 24])
    speed_list += [speed]


# average speed
print("The mouse's average speed is: ", np.mean(speed_list))


# standard deviation
print("The standard deviation is: ", np.std(speed_list))


# standard error of the mean
print("The standard error of the mean is: ", np.std(
    speed_list) / np.sqrt(np.size(speed_list)))


# sort array in ascending order
sorted_speed_list = np.sort(speed_list)


# find the max speed first.
max_speed = np.max(sorted_speed_list)
min_speed = np.min(sorted_speed_list)

print("Also found MAX SPEED and MINIMUM SPEED")


# %% Gaussian filtering
print("\n\n\n=====> Performed guassian filtering in the speed list. <===== \n\n\n")

# Perform guassian filtering in the speed list
speed_list = np.array(speed_list)

# Stacking the X and Y coordinates columns vertically
xy = np.vstack([dlc_data.iloc[:, 18], range(len(dlc_data.iloc[:, 19]))])

# Applying gaussian filtering
z = gaussian_kde(xy)(xy)
fig1, ax = plt.subplots(1, 1)

# Plot
ax.scatter(dlc_data.iloc[:, 18], dlc_data.iloc[:, 19], c=z, s=1)
plt.show()


# %% Creating quartiles

# Split the dlc_data array into 4 equal parts (quartiles)
# Specify how many parts
set_quartiles = 4
quartiles = np.array(dlc_data)
quartiles = np.array_split(quartiles, set_quartiles)

# %% Plotting

'''Create a plot with N number of subplots showing the different quartiles of
of the dlc_data and apply guassia smoothing to each subplot'''


plotRows = 2
plotColumns = 2

# Creating 4 (or N) subplots and unpacking the output array immediately
# Perform guassian filtering in the data

fig2, axs = plt.subplots(plotRows, plotColumns)
axes_list = [axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]]


for ax, i in zip(axes_list, range(set_quartiles)):

    # filtering for each quartile
    xy = np.vstack([quartiles[i][:, 18], range(len(quartiles[i][:, 19]))])
    z = gaussian_kde(xy)(xy)

    # plotting for each quartile
    ax.scatter(quartiles[i][:, 18], quartiles[i][:, 19], c=z, s=1)

print("\n\n\n=====> Plotting <===== \n\n\n")
plt.tight_layout()


# %% Find the timestamps of ca detection
print("\n\n\n=====> Finding the timestamps of ca detection <===== \n\n\n")


# See where we have the first ca occurnace and print it's timestamp
calcium_detection_times = beh_data.drop_duplicates(subset=['0.3'])


detect_time = l_search(beh_data.iloc[:, 7], 1)

# Subtracting the first time of ca detection from all the previous times
calcium_detection_times.iloc[:,
                             0] = calcium_detection_times.iloc[:, 0] - detect_time
calcium_detection_times.drop_duplicates(
    subset=None, keep='first', inplace=True, ignore_index=True)


# %% append ca times to dlc_data df


dlc_data.iloc[:, 24] = dlc_data.iloc[:, 24] - detect_time


# %% Phase detection
print("\n\n\n=====> Done with Phase detection. Check CALCIUM df. <===== \n\n\n")

# prep the df
# change the names

calcium_detection_times[calcium_detection_times.shape[1]] = 0
calcium_detection_times.rename(columns={"8": "ROI"}, inplace=True)

calcium = calcium_detection_times
calcium[8][calcium['Initiation'] == True] = 'Initiation'
calcium[8][calcium['Incorrect'] == True] = 'Incorrect'
calcium[8][calcium['Reward'] == True] = 'Reward'
calcium[8][(calcium['Initiation'] == False) &
           (calcium['Reward'] == False)] = 'Task'
