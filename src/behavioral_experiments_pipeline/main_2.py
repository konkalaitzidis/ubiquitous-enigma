#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 13:15:00 2022

@author: konstantinos kalaitzidis

A module-level docstring

Pipeline to analyze behavioral, tracking, and calcium imagery data.

"""
# %% Importing Packages and Libraries -> Run 1

import warnings
# from time import sleep
from scipy.spatial.distance import euclidean
# import scipy.interpolate as interp
import time
# import math
# from sklearn.preprocessing import normalize
# from scipy import stats
from sklearn import preprocessing
import seaborn as sns
from linear_search import l_search
# from scipy.stats import gaussian_kde
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import calculate_distance
import seaborn as sns
from dataset_info import info
pd.options.mode.chained_assignment = None  # default='warn'

# %% Data Preparation -> Run 2

print("\n\n\n=====> Data Preparation... <===== \n\n\n")


# Preparing behavioral data file
beh_data_path = input("Insert path of behavioral data file here: ")
beh_data = pd.read_csv(beh_data_path, header=None)


# Adding column names
beh_data = beh_data.rename(columns={
    0: 'Time', 1: 'Trial_Number',
    2: 'Reward', 3: 'Frame_Number', 4: 'Central_Zone',
    5: 'L_Zone', 6: 'R_Zone', 7: 'fip_frame'})


# Preparing deep lab cut file
dlc_data_path = input("Insert path of dlc data file here: ")
dlc_data = pd.read_hdf(dlc_data_path)


# Add a column named "Time"
dlc_data[dlc_data.shape[1]] = 0
dlc_data = dlc_data.rename(columns={24: 'Time'})


# Reading the h5 file that contains the deeplabcut and calcium imaging
pathname = input("Insert path of h5 file here: ")

with pd.HDFStore(pathname) as hdf:
    # This prints a list of all group names:
    print("Reading the h5 file that contains the deeplabcut and calcium imaging data...")

h5_file = pd.read_hdf(pathname, key="/meta")


# Removing the date part from the datetime values to keep only seconds
beh_data['Time'] = pd.to_datetime(beh_data['Time'])
beh_data.iloc[:, 0] = beh_data.iloc[:, 0] - beh_data.iloc[0, 0]
beh_data['Time'] = beh_data.iloc[:, 0].dt.total_seconds()


# Adding time to the dlc_data file with the according step.
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

# %% Creating behavioral file for left-turn trials -> Run 2

print("Behavioral data file size is: ", beh_data.size, " and type is: ", type(beh_data))
print("Behavioral data file size is: ",
      beh_data.iloc[:, 0].size, " and type is: ", type(beh_data.iloc[:, 0]))
print("Behavioral data file size is: ",
      beh_data.iloc[:, 1].size, " and type is: ", type(beh_data.iloc[:, 1]))
print("Behavioral data file size is: ",
      beh_data.iloc[:, 2].size, " and type is: ", type(beh_data.iloc[:, 2]))
print("Behavioral data file size is: ",
      beh_data.iloc[:, 3].size, " and type is: ", type(beh_data.iloc[:, 3]))
print("Behavioral data file size is: ",
      beh_data.iloc[:, 4].size, " and type is: ", type(beh_data.iloc[:, 4]))
print("Behavioral data file size is: ",
      beh_data.iloc[:, 5].size, " and type is: ", type(beh_data.iloc[:, 5]))
# print("Behavioral data file size is: ",
#       beh_data.iloc[:, 6].size, " and type is: ", type(beh_data.iloc[:, 6]))

# create new dataframe
init_rew_beh = pd.concat([beh_data.iloc[:, 0], beh_data.iloc[:, 1], beh_data.iloc[:, 2], beh_data.iloc[:, 3], beh_data.iloc[:, 4],
                         beh_data.iloc[:, 5]], axis=1)
print("Initiation to Reward Behavioral data file size is: ",
      init_rew_beh.size, " and type is: ", type(init_rew_beh))


init_rew_beh["Central_Zone"][init_rew_beh["Central_Zone"] == True] = 1
init_rew_beh["L_Zone"][init_rew_beh["L_Zone"] == True] = 1


# %% merging truth values from other columns to central zone -> Run 3


init_rew_beh["Central_Zone"][init_rew_beh["Central_Zone"] == True] = 1
init_rew_beh["Central_Zone"][init_rew_beh["Central_Zone"] == False] = 2
init_rew_beh["L_Zone"][init_rew_beh["L_Zone"] == False] = 2
init_rew_beh["L_Zone"][init_rew_beh["L_Zone"] == True] = 3

# replace the 2 values in C_zone with 3 if L_Zone is 3
init_rew_beh["Central_Zone"][init_rew_beh["L_Zone"] == 3] = 3

# %% Adding index column to dlc file -> Run 4

dlc_data['index_col'] = dlc_data.index

# %% Find and plot only left turn coordinates for correct trials

col0 = dlc_data.iloc[:-1, 18]  # x coordinates
col1 = dlc_data.iloc[:-1, 19]  # y coordinates
col2 = dlc_data.iloc[:-1, 24]  # time
# col3 = speed_df.iloc[:-1, 0]  # speed values
coords_df = pd.concat([col0, col1, col2], axis=1)  # x, y coordinates dataframe and time
# rename column names
# coords_df = coords_df.rename(columns={
#     0: 'x', 1: 'y'})

print("X coordinates: ", col0.size, "& ", col0.isnull().sum(), " missing values")
print("Y coordinates: ", col1.size, "& ", col1.isnull().sum(), " missing values")
print("Time: ", col2.size, "& ", col2.isnull().sum(), " missing values")
# print("Speed values: ", col3.size, "& ", col3.isnull().sum(), " missing values")
print("Coords_df: ", coords_df.size, "& ", coords_df.isnull().sum(), " missing values")


# left turning coordinates x & y
lcx = coords_df.iloc[:, 0].where(coords_df.iloc[:, 0] < 700)
lcy = coords_df.iloc[:, 1].where((coords_df.iloc[:, 1] < 700) & (coords_df.iloc[:, 1] > 200))

# replace missing values with the 0
print("X coordinates: ", lcx.size, "& ", lcx.isnull().sum(), " missing values")
print("Y coordinates: ", lcy.size, "& ", lcy.isnull().sum(), " missing values")

lcy = lcy.fillna(0)
print("Y coordinates: ", lcy.size, "& ", lcy.isnull().sum(), " missing values")
# drop all zeros

coords_df = pd.concat([lcx, lcy, col2], axis=1)  # col3
coords_df.isnull().sum()

coords_df = coords_df[coords_df.iloc[:, 1] != 0]
coords_df['index_col'] = coords_df.index


plt.scatter(coords_df.iloc[:, 0], coords_df.iloc[:, 1], s=0.01)
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


# %% Find speed values for all left-turn coordinates


print("\n\n\n=====> Processing... <===== \n\n\n")

'''Calculate the average speed, the standard deviation, and the standard error of the mean
of the mouse on the behavioral file according to the coordinates on the dlc file'''


# List where speed values will be stored
speed_list = []


# Find all the speeds of the mouse
for index, row in dlc_data.iterrows():

    # control if to exit the function
    if dlc_data.iat[index, -1] == dlc_data.iat[-1, -1]:
        print("All speed values have been stored in list successfully.")
        break

    x1 = dlc_data.iat[index, 0]
    x2 = dlc_data.iat[index+1, 1]
    y1 = dlc_data.iat[index, 0]
    y2 = dlc_data.iat[index + 1, 1]

    distance = calculate_distance.dist_calc(x1, x2, y1, y2)
    speed = distance / (dlc_data.iat[index+1, -1]-dlc_data.iat[index, -1])
    speed_list += [speed]

print("The number of instant speeds is:", len(speed_list))

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


# %% Finding the speed of the mouse for Initiation-Reward trials -> Run 4

speed_list = pd.Series(speed_list)  # speed values of mice for whole session
speed_time = dlc_data.iloc[:, 24]  # time values for whole session


# Speed dataframe
speed_df = pd.DataFrame({'SPEED': speed_list, 'TIME': speed_time}).dropna()


print("Line graph: ")
plt.plot(speed_df.iloc[:, 1], speed_df.iloc[:, 0])
plt.xlabel("Session t")
plt.ylabel("Speed")
plt.show()


# %%

col3 = speed_df.iloc[:-1, 0]  # speed values
coords_df['Speed Values'] = col3  # x, y coordinates dataframe and time
coords_df = coords_df.dropna()

left_turn_time = coords_df.iloc[:, 2]
left_turn_speeds = coords_df.iloc[:, 4]
print("Line graph: ")
plt.plot(left_turn_time, left_turn_speeds)
plt.xlabel("Session t")
plt.ylabel("Speed")
plt.show()


# missing data
ax = plt.axes()
sns.heatmap(coords_df.isna().transpose(), cbar=False, ax=ax)
coords_df.isnull().sum()


# %% extract all correct left-turn trials -> Run 7

warnings.simplefilter(action='ignore', category=FutureWarning)


# def start_and_end_time_of_trial(correct_trial_DF, start_time, end_time, trial_number, start_frame, end_frame):
#     correct_trial_DF = correct_trial_DF.append(
#         {'Trial Number': trial_number, 'Start time': start_time, 'End time': end_time, 'Start Frame': start_frame, 'End Frame': end_frame}, ignore_index=True)
#     return correct_trial_DF


# take the time values from cz 1 -> 3
trial_DF = pd.DataFrame([], columns=['Timestamps', 'Value'])
correct_trial_DF = pd.DataFrame()
end_time = 0
start_time = -1
correct_sequence = False
trial_number = 0
Cz2 = False
Cz3 = False
trial_list = []
start_frame = 0
end_frame = 0


for index, row, in init_rew_beh.iterrows():  # for every row in df

    # if value of column Central Zone is 1 (True)
    if init_rew_beh.iloc[index, 4] == 1 and Cz2 == False:

        if start_time == -1:  # if start_time hasnt been found
            start_time = init_rew_beh.iloc[index, 0]  # start time is the time in index row
        start_frame = init_rew_beh.iloc[index, 3]

        # either way add the index row in a new df
        trial_DF = trial_DF.append(
            {'Timestamps': init_rew_beh.iloc[index, 0], 'Value': init_rew_beh.iloc[index, 4], 'Frame Number': init_rew_beh.iloc[index, 3]}, ignore_index=True)
        correct_sequence = True

    elif init_rew_beh.iloc[index, 4] == 2 and correct_sequence == True and Cz3 == False:
        trial_DF = trial_DF.append(
            {'Timestamps': init_rew_beh.iloc[index, 0], 'Value': init_rew_beh.iloc[index, 4], 'Frame Number': init_rew_beh.iloc[index, 3]}, ignore_index=True)
        # correct_sequence = True
        Cz2 = True

    elif init_rew_beh.iloc[index, 4] == 3 and correct_sequence == True:
        trial_DF = trial_DF.append(
            {'Timestamps': init_rew_beh.iloc[index, 0], 'Value': init_rew_beh.iloc[index, 4], 'Frame Number': init_rew_beh.iloc[index, 3]}, ignore_index=True)
        # correct_sequence = True

        if init_rew_beh.iloc[index + 1, 4] == 2:
            correct_sequence = False
            end_frame = init_rew_beh.iloc[index, 3]
            trial_number += 1
            end_time = init_rew_beh.iloc[index, 0]

            correct_trial_DF = correct_trial_DF.append(
                {'Trial Number': trial_number, 'Start Frame': start_frame, 'End Frame': end_frame, 'Start time': start_time, 'End time': end_time}, ignore_index=True)

            # reset variables
            Cz2 = False
            start_time = -1
            Cz3 = True

correct_trial_DF = correct_trial_DF.iloc[:, 0:3].astype(int)
print(correct_trial_DF)


# drop the last row
correct_trial_DF = correct_trial_DF[:-1]
# coords_df.drop(coords_df.index[correct_trial_DF.iloc[-1, 1]:len(coords_df)])


# for index, row in correct_trial_DF.iterrows():
#     correct_trial_DF.iloc[index, 1].drop([where(correct_trial_DF.iloc[index, 1] > coords_df.iloc[index, 3])], axis=0)


# correct_trial_DF.drop([correct_trial_DF.where(correct_trial_DF.iloc[:, 1] >= coords_df.iloc[:, -2])])
# df.drop(df[df.scrrect_trial_DF = ore < 50].index, inplace=True)


# %%
trial_count = 0
new_dlc_data = pd.DataFrame()

for index, row in correct_trial_DF.iterrows():

    for index in range(correct_trial_DF.iloc[-1, 1]):
        if coords_df.iloc[index, 3] >= correct_trial_DF.iloc[trial_count, 1] and coords_df.iloc[index, 3] <= correct_trial_DF.iloc[trial_count, 2]:
            new_dlc_data = new_dlc_data.append(coords_df.iloc[index, :])
    trial_count += 1

    print(new_dlc_data)


# %% Plot distance and speed -> Run 10

# slice a dataset in bins
set_bins = len(correct_trial_DF)
coords_quartiles = np.array(new_dlc_data)
coords_quartiles = np.array_split(coords_quartiles, set_bins)

average_speed_list = []
for index in range(set_bins):
    average_speed = np.median(coords_quartiles[index][:, 2])
    average_speed_list += [average_speed]
    # print("The mouse's average speed is for bin", index, " is: ", average_speed_list[index])
    index += 1
print("Average speed for each bin: \n", average_speed_list, "\n\n")


# eucledean distance
bin_distance_list = []
total_distance = 0
total_distance_list = []
for index in range(set_bins):
    bin_distance = euclidean(coords_quartiles[index][:, 0], coords_quartiles[index][:, 1])
    bin_distance_list += [bin_distance]
    total_distance = total_distance + bin_distance_list[index]
    total_distance_list += [total_distance]
print("Calculated distance each bin: \n", bin_distance_list)
print("Total distance: ", total_distance)


# normalize
total_distance_list = np.array(total_distance_list)
average_speed_list = np.array(average_speed_list)

min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(total_distance_list.reshape(-1, 1))

min_max_scaler = preprocessing.MinMaxScaler()
y_scaled = min_max_scaler.fit_transform(average_speed_list.reshape(-1, 1))


# plot
plt.figure(figsize=(7, 3))
print("Plot: \n")
plt.plot(x_scaled, y_scaled)
plt.xlabel("distance")
plt.ylabel("Speed")
plt.show()


# %%
print("Line graph: ")

for index in range(set_bins):
    plt.plot(coords_quartiles[index][:, 2], coords_quartiles[index][:, 4])
    plt.xlabel("Session t")
    plt.ylabel("Speed")
    plt.show()

j = 0
for i, row in correct_trial_DF.iterrows():
    for j in range(correct_trial_DF.iloc[j, 1], correct_trial_DF.iloc[j, 1]):

        # find me the euclidean distance for this trial
        bin_distance = euclidean(coords_quartiles[index][:, 0], coords_quartiles[index][:, 1])
        bin_distance_list += [bin_distance]
        total_distance = total_distance + bin_distance_list[index]
        total_distance_list += [total_distance]
        print("Calculated distance each bin: \n", bin_distance_list)
        print("Total distance: ", total_distance)

        # findme the average speeds

        # store
        plt.plot(coords_quartiles[index][:, 2], coords_quartiles[index][:, 4])
        plt.xlabel("Session t")
        plt.ylabel("Speed")
        plt.show()

        j += 1  # increase j

plt.figure(figsize=(7, 3))
plt.plot(x_scaled, new_dlc_data.iloc[2477:5569, 4])
plt.xlabel("Session t")
plt.ylabel("Speed")
plt.show()
