#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 13:15:00 2022

@author: konstantinos kalaitzidis

A module-level docstring

Pipeline to analyze behavioral, tracking, and calcium imagery data.

"""
# %% Importing Packages and Libraries

import scipy.interpolate as interp
import time
import math
from sklearn.preprocessing import normalize
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
# set_quartiles = 4
# quartiles = np.array(dlc_data)
# quartiles = np.array_split(quartiles, set_quartiles)


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
speed_df = pd.DataFrame({'SPEED': speed_list, 'TIME': speed_time})


print("Line graph: ")
plt.plot(speed_df.iloc[:, 1], speed_df.iloc[:, 0])
plt.xlabel("Session t")
plt.ylabel("Speed")
plt.show()

# # Normalization
# x = np.array(speed_df)  # returns a numpy array
# min_max_scaler = preprocessing.MinMaxScaler()
# x_scaled = min_max_scaler.fit_transform(x)
# x_scaled_speed_df = pd.DataFrame(x_scaled)


# # Plotting a line graph - revisit
# print("Line graph: ")
# plt.plot(x_scaled_speed_df.iloc[:, 1], x_scaled_speed_df.iloc[:, 0])
# plt.xlabel("Session Period")
# plt.ylabel("Instantaneous Speed")
# plt.show()


# Find and plot only left turn coordinates

col0 = dlc_data.iloc[:-1, 18]  # x coordinates
col1 = dlc_data.iloc[:-1, 19]  # y coordinates
col2 = dlc_data.iloc[:-1, 24]  # time
col3 = speed_df.iloc[:-1, 0]  # speed values
coords_df = pd.concat([col0, col1, col2, col3], axis=1)  # x, y coordinates dataframe and time
# rename column names
# coords_df = coords_df.rename(columns={
#     0: 'x', 1: 'y'})

print("X coordinates: ", col0.size, "& ", col0.isnull().sum())
print("Y coordinates: ", col1.size, "& ", col1.isnull().sum())
print("Time: ", col2.size, "& ", col2.isnull().sum())
print("Speed values: ", col3.size, "& ", col3.isnull().sum())
print("Coords_df: ", coords_df.size, "& ", coords_df.isnull().sum())


# left turning coordinates x & y
lcx = coords_df.iloc[:, 0].where(coords_df.iloc[:, 0] < 700)
lcy = coords_df.iloc[:, 1].where((coords_df.iloc[:, 1] < 700) & (coords_df.iloc[:, 1] > 200))

# replace missing values with the 0
print("X coordinates: ", lcx.size, "& ", lcx.isnull().sum())
print("Y coordinates: ", lcy.size, "& ", lcy.isnull().sum())

lcy = lcy.fillna(0)
print("Y coordinates: ", lcy.size, "& ", lcy.isnull().sum())
# drop all zeros

coords_df = pd.concat([lcx, lcy, col2, col3], axis=1)
coords_df.isnull().sum()

coords_df = coords_df[coords_df.iloc[:, 1] != 0]


plt.scatter(coords_df.iloc[:, 0], coords_df.iloc[:, 1])
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


left_turn_time = coords_df.iloc[:, 2]
left_turn_speeds = coords_df.iloc[:, 3]
print("Line graph: ")
plt.plot(left_turn_time, left_turn_speeds)
plt.xlabel("Session t")
plt.ylabel("Speed")
plt.show()


# missing data
ax = plt.axes()
sns.heatmap(coords_df.isna().transpose(), cbar=False, ax=ax)
coords_df.isnull().sum()


# slice a dataset in bins
set_bins = 100
coords_quartiles = np.array(coords_df)
coords_quartiles = np.array_split(coords_quartiles, set_bins)
average_speed_list = []
for index in range(set_bins):
    average_speed = np.mean(coords_quartiles[index][:, 3])
    average_speed_list += [average_speed]
    #print("The mouse's average speed is for bin", index, " is: ", average_speed_list[index])
    index = index+1
print("Done")

plt.plot(np.arange(100), average_speed_list)
plt.xlabel("bins")
plt.ylabel("avg speeds")
plt.show()


for index in range(set_bins):
    min_point = min(coords_quartiles[index][:, :1])
    max_point = max(coords_quartiles[index][:, :1])


# find point a
max_x = max(coords_df.iloc[:, 0])
for index, row in coords_df.iterrows():
    if coords_df.iloc[index, 0] == max_x:
        coords_df_y = coords_df.iloc[index, 1]
print(coords_df_y)
point_a = (max_x, coords_df_y)

# find point b
max_y = max(coords_df.iloc[:, 1])
for index, row in coords_df.iterrows():
    if coords_df.iloc[index, 1] == max_y:
        coords_df_x = coords_df.iloc[index, 0]
print(coords_df_x)
point_b = (coords_df_x, max_y)


# distance b/w a and b
d1 = math.dist(point_a, point_b)
# display the result
print(d1)

# find point c
min_x = min(coords_df.iloc[:, 0])
for index, row in coords_df.iterrows():
    if coords_df.iloc[index, 0] == min_x:
        coords_df_y2 = coords_df.iloc[index, 1]
print(coords_df_y2)
point_c = (min_x, coords_df_y2)

d2 = math.dist(point_b, point_c)

total_dist = d1 + d2


# for ax, i in zip(axes_list, range(set_quartiles)):
#
#     # filtering for each quartile
#     xy = np.vstack([quartiles[i][:, 18], range(len(quartiles[i][:, 19]))])
#     z = gaussian_kde(xy)(xy)
#
#     # plotting for each quartile
#     ax.scatter(quartiles[i][:, 18], quartiles[i][:, 19], c=z, s=1)


# Create coordinate bins
''' Phases:
    bin_1 = Initiation
    bin_2 = Initiation -> Choice
    bin_3 = Choice
    bin_4 = Choice -> Reward
    bin_5 = Reward '''


# bin_1 | Initiation
init_x = coords_df.iloc[:, 0].where(coords_df.iloc[:, 0] > 500)
init_y = coords_df.iloc[:, 1].where((coords_df.iloc[:, 1] < 300) & (coords_df.iloc[:, 1] > 0))
# init_df = pd.concat([init_x, init_y], axis=1)

# min_max_scaler = preprocessing.MinMaxScaler()
# x_scaled = min_max_scaler.fit_transform(init_x)

# min_max_scaler = preprocessing.MinMaxScaler()
# x_scaled = min_max_scaler.fit_transform(init_y)

plt.scatter(init_x, init_y, s=5, c='#800000')
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


# bin_2 | Initiation -> Choice
init_ch_x = coords_df.iloc[:, 0].where((coords_df.iloc[:, 0] > 280) & (coords_df.iloc[:, 0] < 500))
init_ch_y = coords_df.iloc[:, 1].where((coords_df.iloc[:, 1] < 600) & (coords_df.iloc[:, 1] > 300))
# init_ch_df = pd.concat([init_ch_x, init_ch_y], axis=1)

# min_max_scaler = preprocessing.MinMaxScaler()
# x_scaled = min_max_scaler.fit_transform(init_ch_df)

plt.scatter(init_ch_x, init_ch_y, s=5, c='#8B0000')
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


# bin_3 | Choice
ch_x = coords_df.iloc[:, 0].where((coords_df.iloc[:, 0] > 150) & (coords_df.iloc[:, 0] < 280))
ch_y = coords_df.iloc[:, 1].where(coords_df.iloc[:, 1] > 600)
# ch_df = pd.concat([ch_x, ch_y], axis=1)

# min_max_scaler = preprocessing.MinMaxScaler()
# x_scaled = min_max_scaler.fit_transform(ch_df)

plt.scatter(ch_x, ch_y, s=5, c='#B22222')
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


# bin_4 | Choice -> Reward
ch_rew_x = coords_df.iloc[:, 0].where((coords_df.iloc[:, 0] > 130) & (coords_df.iloc[:, 0] < 240))
ch_rew_y = coords_df.iloc[:, 1].where((coords_df.iloc[:, 1] < 600) & (coords_df.iloc[:, 1] > 350))
# ch_rew_df = pd.concat([ch_rew_x, ch_rew_y], axis=1)

# min_max_scaler = preprocessing.MinMaxScaler()
# x_scaled = min_max_scaler.fit_transform(ch_rew_df)

plt.scatter(ch_rew_x, ch_rew_y, s=5, c='#DC143C')
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


# bin_5 | Reward
rew_x = coords_df.iloc[:, 0].where((coords_df.iloc[:, 0] > 130) & (coords_df.iloc[:, 0] < 240))
rew_y = coords_df.iloc[:, 1].where((coords_df.iloc[:, 1] < 350) & (coords_df.iloc[:, 1] > 240))
# rew_df = pd.concat([rew_x, rew_y], axis=1)


plt.scatter(rew_x, rew_y, s=5, c='#FF0000')
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


# plotRows = 3
# plotColumns = 2


# # Creating 5 subplots of phase bins
# fig2, axs = plt.subplots(plotRows, plotColumns)
# axes_list = [axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1], axs[2, 0]]

# set_bins = 5
# bin_1 = np.array(init_df)
# bin_2 = np.array(init_ch_df)
# bin_3 = np.array(ch_df)
# bin_4 = np.array(ch_rew_df)
# bin_5 = np.array(rew_df)

# # # have subplot titles
# # for i in range(set_bins):
# #     axes_list.title.set_text('% plot', i)

# # a list of all numpy arrays
# bins = [bin_1, bin_2, bin_3, bin_4, bin_5]

# # min_max_scaler = preprocessing.MinMaxScaler()

# for axs, i in zip(axes_list, range(set_bins)):
#     # normalizing data
#     # bins[i] = min_max_scaler.fit_transform(bins[i])
#     print(f"Minimum value in data is: {np.min(bins[i])}")
#     print(f"Maximum value in data is: {np.max(bins[i])}")
#     axs.scatter(bins[i][:, 0], bins[i][:, 1], s=0.5)

# print("\n\n\n=====> Plotting <===== \n\n\n")
# plt.tight_layout()

# %% Find the linear and angualr velocity of the animal when it turns left during init -> reward trial


'''A kernel density estimate (KDE) plot is a method for visualizing the distribution of observations in a dataset, analogous to a histogram. KDE represents the data using a continuous probability density curve in one or more dimensions.'''
sns.kdeplot(x=lcx, y=lcy.where(lcy > 0))

# missing data
ax = plt.axes()
sns.heatmap(coords_df.isna().transpose(), cbar=False, ax=ax)
coords_df.isnull().sum()


# calculate distnace only for left-turn coordinates


# get the start time
st = time.time()

# List where speed values will be stored
left_speed_list = []
left_turn_distance_list = []
total_left_turn_distance_list = []
total_left_turn_distance = 0

# Find all the speeds of the mouse during left turn trajectory
for index, row in coords_df.iloc[:, :1].iterrows():

    # left turn only

    # control if to exit the function
    if coords_df.iat[index, 1] == coords_df.iat[-1, 1]:
        print("All speed values have been stored in list successfully.")
        break

    x1 = coords_df.iat[index, 0]
    x2 = coords_df.iat[index+1, 0]
    y1 = coords_df.iat[index, 1]
    y2 = coords_df.iat[index + 1, 1]

    left_turn_distance = calculate_distance.dist_calc(x1, x2, y1, y2)
    left_speed = left_turn_distance / (coords_df.iat[index+1, 2]-coords_df.iat[index, 2])

    total_left_turn_distance = total_left_turn_distance + left_turn_distance

    total_left_turn_distance_list += [total_left_turn_distance]

    left_turn_distance_list += [left_turn_distance]
    # print(left_turn_distance_list[index])
    left_speed_list += [left_speed]

print(len(left_turn_distance_list))
print(len(left_speed_list))  # y
print(total_left_turn_distance)


# left_turn_distance_list = np.array(left_turn_distance_list)
# left_speed_list = np.array(left_speed_list)

# total_left_turn_distance_list = np.array(total_left_turn_distance_list).reshape(-1, 1)
# left_speed_list = np.array(left_speed_list).reshape(-1, 1)
# # normalize speed_list and coords

# min_max_scaler = preprocessing.MinMaxScaler()
# x_scaled = min_max_scaler.fit_transform(total_left_turn_distance_list)

# min_max_scaler = preprocessing.MinMaxScaler()
# x_scaled = min_max_scaler.fit_transform(left_speed_list)

plt.plot(total_left_turn_distance_list, left_speed_list)
plt.xlabel("distance")
plt.ylabel("Speed")
plt.show()

# total_distance = np.array(total_distance)

# x_axis = np.linspace(0, 1, total_left_turn_distance)
# y_axis = np.linspace(0, 1, left_speed_list)

# plt.plot(x_axis, y_axis)
# plt.show()

# !sns.displot(data=coords_df.iloc[:, :1], x=coords_df.iloc[:, 0], y=coords_df.iloc[:, 1].where(lcy > 0), kind="kde")
# plt.plot(range(74291), left_turn_distance)

# get the end time
et = time.time()

# get the execution time
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')


# average speed
print("The mouse's average speed is: ", np.mean(speed_list))

# time in relation with speed
plt.plot(coords_df.iloc[:-1, 2], speed_list)

# # find point a
# max_x = max(coords_df.iloc[:, 0])
# for index, row in coords_df.iterrows():
#     if coords_df.iloc[index, 0] == max_x:
#         coords_df_y = coords_df.iloc[index, 1]
# print(coords_df_y)
# point_a = (max_x, coords_df_y)

# # find point b
# max_y = max(coords_df.iloc[:, 1])
# for index, row in coords_df.iterrows():
#     if coords_df.iloc[index, 1] == max_y:
#         coords_df_x = coords_df.iloc[index, 0]
# print(coords_df_x)
# point_b = (coords_df_x, max_y)


# # distance b/w a and b
# d1 = math.dist(point_a, point_b)
# # display the result
# print(d1)

# # find point c
# min_x = min(coords_df.iloc[:, 0])
# for index, row in coords_df.iterrows():
#     if coords_df.iloc[index, 0] == min_x:
#         coords_df_y2 = coords_df.iloc[index, 1]
# print(coords_df_y2)
# point_c = (min_x, coords_df_y2)

# d2 = math.dist(point_b, point_c)

# total_dist = d1 + d2


# x_inter = interp.interp1d(np.arange(coords_df.iloc[:, 2].size), coords_df.iloc[:, 2])
# x_ = x_inter(np.linspace(0, coords_df.iloc[:, 2].size-1, left_speed_list.size))
# print(len(x_), len(left_speed_list))
# plt.plot(x_, left_speed_list)
# plt.xlabel("time")
# plt.ylabel("Speed")
# plt.show()

# sns.displot(data=coords_df.iloc[:, 2], x=total_left_turn_distance_list, kind="kde")


# n = 12
# a = np.arange(n)
# x = 2**a
# y = np.random.rand(n)

# fig = plt.figure(1, figsize=(7, 7))
# ax1 = fig.add_subplot(211)
# ax2 = fig.add_subplot(212)

# ax1.plot(x, y)
# ax1.xaxis.set_ticks(x)


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
