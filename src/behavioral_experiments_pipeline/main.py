#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 13:15:00 2022

@author: konstantinos kalaitzidis

A module-level docstring

This is the a draft custom-made pipeline to analyze behavioral, tracking, and
calcium imagery data.

In collaboration with Thodoris Tamiolakis
"""
# %% Importing Packages and Libraries

from scipy import stats
from sklearn import preprocessing
import seaborn as sns
from linear_search import l_search
from scipy.stats import gaussian_kde
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import calculate_distance
from dataset_info import info
pd.options.mode.chained_assignment = None  # default='warn'


# %% Data Preparation

print("\n\n\n=====> Data Preparation... <===== \n\n\n")


# Preparing behavioral data file
beh_data_path = input("Insert path of behavioral data file here: ")
beh_data = pd.read_csv(beh_data_path, header=None)


# Adding column names
beh_data = beh_data.rename(columns={
    0: 'Time', 1: 'Choice',
    2: 'Init-Reward', 4: 'Initiation',
    5: 'Incorrect', 6: 'Reward', 7: 'CA_Signals'})


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


# %% Understanding our datasets

print("\n\n\n=====> Understanding our datasets <===== \n\n\n")


datafile_names = {'behavioral data': beh_data,
                  'tracking data': dlc_data, 'h5': h5_file}
info(datafile_names)


# %% Store processed data

# TODO if requested in the future


# %% Processing

print("\n\n\n=====> Processing... <===== \n\n\n")

'''Calculate the average speed, the standard deviation, and the standard error of the mean
of the mouse on the behavioral file according to the coordinates on the dlc file'''


# List where speed values will be stored
speed_list = []
# =============================================================================
#
#
# # Find all the speeds of the mouse
# for index, row in dlc_data.iterrows():
#
#     # control if to exit the function
#     if dlc_data.iat[index, 24] == dlc_data.iat[-1, 24]:
#         print("All speed values have been stored in list successfully.")
#         break
#
#     x1 = dlc_data.iat[index, 18]
#     x2 = dlc_data.iat[index+1, 18]
#     y1 = dlc_data.iat[index, 19]
#     y2 = dlc_data.iat[index + 1, 19]
#
#     distance = calculate_distance.dist_calc(x1, x2, y1, y2)
#     speed = distance / (dlc_data.iat[index+1, 24]-dlc_data.iat[index, 24])
#     speed_list += [speed]
#
#
# # average speed
# print("The mouse's average speed is: ", np.mean(speed_list))
#
#
# # standard deviation
# print("The standard deviation is: ", np.std(speed_list))
#
#
# # standard error of the mean
# print("The standard error of the mean is: ", np.std(
#     speed_list) / np.sqrt(np.size(speed_list)))
#
#
# # sort array in ascending order
# sorted_speed_list = np.sort(speed_list)
#
#
# # find the max speed first.
# max_speed = np.max(sorted_speed_list)
# min_speed = np.min(sorted_speed_list)
#
# print("Also found MAX SPEED and MINIMUM SPEED")
#
# =============================================================================

# %% Gaussian filtering

# =============================================================================
# print("\n\n\n=====> Performed guassian filtering in the speed list. <===== \n\n\n")
#
#
# # Perform guassian filtering in the speed list
# speed_list = np.array(speed_list)
#
#
# # Stacking the X and Y coordinates columns vertically
# xy = np.vstack([dlc_data.iloc[:, 18], range(len(dlc_data.iloc[:, 19]))])
#
#
# # Applying gaussian filtering
# z = gaussian_kde(xy)(xy)
# fig1, ax = plt.subplots(1, 1)
#
#
# # Plot
# ax.scatter(dlc_data.iloc[:, 18], dlc_data.iloc[:, 19], c=z, s=1)
# plt.show()
#
# =============================================================================

# %% Creating quartiles

# Split the dlc_data array into 4 equal parts (quartiles)
# Specify how many parts
set_quartiles = 4
quartiles = np.array(dlc_data)
quartiles = np.array_split(quartiles, set_quartiles)


# %% Plotting

# =============================================================================
# '''Create a plot with N number of subplots showing the different quartiles of
# of the dlc_data and apply guassia smoothing to each subplot'''
#
#
# plotRows = 2
# plotColumns = 2
#
# # Creating 4 (or N) subplots and unpacking the output array immediately
# # Perform guassian filtering in the data
#
# fig2, axs = plt.subplots(plotRows, plotColumns)
# axes_list = [axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]]
#
#
# for ax, i in zip(axes_list, range(set_quartiles)):
#
#     # filtering for each quartile
#     xy = np.vstack([quartiles[i][:, 18], range(len(quartiles[i][:, 19]))])
#     z = gaussian_kde(xy)(xy)
#
#     # plotting for each quartile
#     ax.scatter(quartiles[i][:, 18], quartiles[i][:, 19], c=z, s=1)
#
#
# print("\n\n\n=====> Plotting <===== \n\n\n")
# plt.tight_layout()
#
# =============================================================================
# %% Finding the speed of the mouse for Initiation-Reward trials

speed_list = pd.Series(speed_list)  # speed values of mice for whole session
speed_time = dlc_data.iloc[:, -1]  # time values for whole session


# Speed dataframe
speed_df = pd.DataFrame({'TIME': speed_time})
speed_df['SPEED'] = pd.Series(speed_list)


# Normalization
x = speed_df.values  # returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
speed_df = pd.DataFrame(x_scaled)


# Plotting a line graph
print("Line graph: ")
plt.plot(speed_df[0], speed_df[1])
plt.xlabel("Session Period")
plt.ylabel("Instantaneous Speed")
plt.show()


# Find and plot only left turn coordinates


col0 = dlc_data.iloc[:, 18]
col1 = dlc_data.iloc[:, 19]
coords_df = pd.concat([col0, col1], axis=1)  # x, y coordinates dataframe

# left turning coordinates x & y
lcx = coords_df.iloc[:, 0].where(coords_df.iloc[:, 0] < 700)
lcy = coords_df.iloc[:, 1].where(coords_df.iloc[:, 1] < 700)

coords_df = pd.concat([lcx, lcy], axis=1)

# x = speed_df.values  # returns a numpy array
# min_max_scaler = preprocessing.MinMaxScaler()
# x_scaled = min_max_scaler.fit_transform(x)
# speed_df = pd.DataFrame(x_scaled)

plt.scatter(lcx, lcy, s=0.05)
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


# Create coordinate bins
''' Phases:
    bin_1 = Initiation > x500 and <y300
    bin_2 = Initiation -> Choice
    bin_3 = Choice
    bin_4 = Choice -> Reward
    bin_5 = Reward '''


# bin_1 | Initiation
init_x = coords_df.iloc[:, 0].where(coords_df.iloc[:, 0] > 500)
init_y = coords_df.iloc[:, 1].where(coords_df.iloc[:, 1] < 300)
init_df = pd.concat([init_x, init_y], axis=1)

min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(init_df)

plt.scatter(init_x, init_y, s=5, c='#800000')
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


# bin_2 | Initiation -> Choice
init_ch_x = coords_df.iloc[:, 0].where((coords_df.iloc[:, 0] > 280) & (coords_df.iloc[:, 0] < 500))
init_ch_y = coords_df.iloc[:, 1].where((coords_df.iloc[:, 1] < 600) & (coords_df.iloc[:, 1] > 300))
init_ch_df = pd.concat([init_ch_x, init_ch_y], axis=1)

min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(init_ch_df)

plt.scatter(init_ch_x, init_ch_y, s=5, c='#8B0000')
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


# bin_3 | Choice
ch_x = coords_df.iloc[:, 0].where((coords_df.iloc[:, 0] > 150) & (coords_df.iloc[:, 0] < 280))
ch_y = coords_df.iloc[:, 1].where(coords_df.iloc[:, 1] > 600)
ch_df = pd.concat([init_ch_x, init_ch_y], axis=1)

min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(ch_df)

plt.scatter(ch_x, ch_y, s=5, c='#B22222')
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


# bin_4 | Choice -> Reward
ch_rew_x = coords_df.iloc[:, 0].where((coords_df.iloc[:, 0] > 130) & (coords_df.iloc[:, 0] < 240))
ch_rew_y = coords_df.iloc[:, 1].where((coords_df.iloc[:, 1] < 600) & (coords_df.iloc[:, 1] > 350))
ch_rew_df = pd.concat([init_ch_x, init_ch_y], axis=1)

min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(ch_rew_df)

plt.scatter(ch_rew_x, ch_rew_y, s=5, c='#DC143C')
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


# bin_5 | Reward
rew_x = coords_df.iloc[:, 0].where((coords_df.iloc[:, 0] > 130) & (coords_df.iloc[:, 0] < 240))
rew_y = coords_df.iloc[:, 1].where((coords_df.iloc[:, 1] < 350) & (coords_df.iloc[:, 1] > 240))
rew_df = pd.concat([init_ch_x, init_ch_y], axis=1)

min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(rew_df)

plt.scatter(rew_x, rew_y, s=5, c='#FF0000')
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


#bins = np.array_split(bins, set_bins)


plotRows = 3
plotColumns = 2


# Creating 5 subplots of phase bins
fig2, axs = plt.subplots(plotRows, plotColumns)
axes_list = [axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1], axs[2, 0]]

set_bins = 5
bin_1 = np.array(init_df)
bin_2 = np.array(init_ch_df)
bin_3 = np.array(ch_df)
bin_4 = np.array(ch_rew_df)
bin_5 = np.array(rew_df)

# a list of all numpy arrays
bins = [bin_1, bin_2, bin_3, bin_4, bin_5]

for ax, i in zip(axes_list, range(set_bins)):

    # # filtering for each quartile
    # xy = np.vstack([bins[i][:, 18], range(len(bins[i][:, 19]))])
    # z = gaussian_kde(xy)(xy)

    # plotting for each quartile
    ax.scatter(bins[i][:, 0], bins[i][:, 1], s=0.05)
    i = i+1


print("\n\n\n=====> Plotting <===== \n\n\n")
plt.tight_layout()


# %% Find the timestamps of ca detection

print("\n\n\n=====> Finding the timestamps of ca detection <===== \n\n\n")


# See where we have the first ca occurnace and print it's timestamp
calcium_detection_times = beh_data.drop_duplicates(subset=['CA_Signals'])

ca = beh_data.iloc[:, 7]  # column with calcium signals
t = beh_data.iloc[:, 0]  # column with time
search_for = 1  # first instance of calcium signal

detect_time = l_search(ca, search_for, t)


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
calcium = calcium_detection_times.rename(columns={8: "ROI"})

calcium["ROI"][calcium['Initiation'] == True] = 'Initiation'
calcium["ROI"][calcium['Incorrect'] == True] = 'Incorrect'
calcium["ROI"][calcium['Reward'] == True] = 'Reward'
calcium["ROI"][(calcium['Initiation'] == False) &
               (calcium['Reward'] == False)] = 'Task'
