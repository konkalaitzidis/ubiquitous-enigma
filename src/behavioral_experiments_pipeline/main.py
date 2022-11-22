#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 13:15:00 2022

@author: konstantinos kalaitzidis

A module-level docstring

Pipeline to analyze behavioral, tracking, and calcium imagery data.

"""
# %% Importing Packages and Libraries

import warnings
from time import sleep
from scipy.spatial.distance import euclidean
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

# print("\n\n\n=====> Performed guassian filtering in the speed list. <===== \n\n\n")


# # Perform guassian filtering in the speed list
# speed_list = np.array(speed_list)


# # Stacking the X and Y coordinates columns vertically
# xy = np.vstack([dlc_data.iloc[:, 18], range(len(dlc_data.iloc[:, 19]))])


# # Applying gaussian filtering
# z = gaussian_kde(xy)(xy)
# fig1, ax = plt.subplots(1, 1)


# # Plot
# ax.scatter(dlc_data.iloc[:, 18], dlc_data.iloc[:, 19], c=z, s=1)
# plt.show()


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

# %%
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


# d1 = 0
# d2 = 0
# quartile_d = 0  # quartile distance
# quartile_d_list = []
# distance_sum = 0
# speed_list = []
# for index in range(set_bins):
#     # d1 = min(coords_quartiles[index][:, 1])
#     # d2 = max(coords_quartiles[index][:, 1])
#     # quartile_d = d2 - d1
#     # quartile_d_list += [quartile_d]
#     # distance_sum = distance_sum + quartile_d
#     # index = index +
#     # List where speed values will be stored
#     # Find all the speeds of the mouse
#     x=0
#     for x in np.nditer(coords_quartiles[x]):

#         # control if to exit the function
#         if coords_quartiles[index][index, 2] == coords_quartiles[index][-1, 2]:
#             print("All speed values have been stored in list successfully.")
#             break

#         x1 = coords_quartiles[x][x, 0]
#         x2 = coords_quartiles[x][x+1, 0]
#         y1 = coords_quartiles[x][x, 1]
#         y2 = coords_quartiles[x][x + 1, 1]

#         distance = calculate_distance.dist_calc(x1, x2, y1, y2)
#         speed = distance / (coords_quartiles[x][x+1, 2] - coords_quartiles[x][x, 2])
#         speed_list += [speed]
#         quartile_d_list += [distance]
#     # break
#     distance_sum = distance_sum + distance

# print("Total Distance is: ", distance_sum)


# plt.plot(np.arange(set_bins), average_speed_list)
# plt.xlabel("bins")
# plt.ylabel("avg speeds")
# plt.show()


# find point a
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


# # bin_1 | Initiation
# init_x = coords_df.iloc[:, 0].where(coords_df.iloc[:, 0] > 500)
# init_y = coords_df.iloc[:, 1].where((coords_df.iloc[:, 1] < 300) & (coords_df.iloc[:, 1] > 0))
# # init_df = pd.concat([init_x, init_y], axis=1)

# # min_max_scaler = preprocessing.MinMaxScaler()
# # x_scaled = min_max_scaler.fit_transform(init_x)

# # min_max_scaler = preprocessing.MinMaxScaler()
# # x_scaled = min_max_scaler.fit_transform(init_y)

# plt.scatter(init_x, init_y, s=5, c='#800000')
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.show()


# # bin_2 | Initiation -> Choice
# init_ch_x = coords_df.iloc[:, 0].where((coords_df.iloc[:, 0] > 280) & (coords_df.iloc[:, 0] < 500))
# init_ch_y = coords_df.iloc[:, 1].where((coords_df.iloc[:, 1] < 600) & (coords_df.iloc[:, 1] > 300))
# # init_ch_df = pd.concat([init_ch_x, init_ch_y], axis=1)

# # min_max_scaler = preprocessing.MinMaxScaler()
# # x_scaled = min_max_scaler.fit_transform(init_ch_df)

# plt.scatter(init_ch_x, init_ch_y, s=5, c='#8B0000')
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.show()


# # bin_3 | Choice
# ch_x = coords_df.iloc[:, 0].where((coords_df.iloc[:, 0] > 150) & (coords_df.iloc[:, 0] < 280))
# ch_y = coords_df.iloc[:, 1].where(coords_df.iloc[:, 1] > 600)
# # ch_df = pd.concat([ch_x, ch_y], axis=1)

# # min_max_scaler = preprocessing.MinMaxScaler()
# # x_scaled = min_max_scaler.fit_transform(ch_df)

# plt.scatter(ch_x, ch_y, s=5, c='#B22222')
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.show()


# # bin_4 | Choice -> Reward
# ch_rew_x = coords_df.iloc[:, 0].where((coords_df.iloc[:, 0] > 130) & (coords_df.iloc[:, 0] < 240))
# ch_rew_y = coords_df.iloc[:, 1].where((coords_df.iloc[:, 1] < 600) & (coords_df.iloc[:, 1] > 350))
# # ch_rew_df = pd.concat([ch_rew_x, ch_rew_y], axis=1)

# # min_max_scaler = preprocessing.MinMaxScaler()
# # x_scaled = min_max_scaler.fit_transform(ch_rew_df)

# plt.scatter(ch_rew_x, ch_rew_y, s=5, c='#DC143C')
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.show()


# # bin_5 | Reward
# rew_x = coords_df.iloc[:, 0].where((coords_df.iloc[:, 0] > 130) & (coords_df.iloc[:, 0] < 240))
# rew_y = coords_df.iloc[:, 1].where((coords_df.iloc[:, 1] < 350) & (coords_df.iloc[:, 1] > 240))
# # rew_df = pd.concat([rew_x, rew_y], axis=1)


# plt.scatter(rew_x, rew_y, s=5, c='#FF0000')
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.show()


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


# '''A kernel density estimate (KDE) plot is a method for visualizing the distribution of observations in a dataset, analogous to a histogram. KDE represents the data using a continuous probability density curve in one or more dimensions.'''
# sns.kdeplot(x=lcx, y=lcy.where(lcy > 0))

# # missing data
# ax = plt.axes()
# sns.heatmap(coords_df.isna().transpose(), cbar=False, ax=ax)
# coords_df.isnull().sum()


# %% create data
# Find and plot only left turn coordinates

col0 = dlc_data.iloc[:-1, 18]  # x coordinates
col1 = dlc_data.iloc[:-1, 19]  # y coordinates
col2 = dlc_data.iloc[:-1, 24]  # time
col3 = speed_df.iloc[:-1, 0]  # speed values
coords_df = pd.concat([col0, col1, col2, col3], axis=1)  # x, y coordinates dataframe and time


print("X coordinates: ", col0.size, "& ", col0.isnull().sum(), " missing values")
print("Y coordinates: ", col1.size, "& ", col1.isnull().sum(), " missing values")
print("Time: ", col2.size, "& ", col2.isnull().sum(), " missing values")
print("Speed values: ", col3.size, "& ", col3.isnull().sum(), " missing values")
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

coords_df = pd.concat([lcx, lcy, col2, col3], axis=1)
coords_df.isnull().sum()

coords_df = coords_df[coords_df.iloc[:, 1] != 0]


plt.scatter(coords_df.iloc[:, 0], coords_df.iloc[:, 1], s=0.01)
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

# %% Plot distance and speed for the whole left turn

# slice a dataset in bins
set_bins = 100
coords_quartiles = np.array(coords_df)
coords_quartiles = np.array_split(coords_quartiles, set_bins)

average_speed_list = []
for index in range(set_bins):
    average_speed = np.mean(coords_quartiles[index][:, 3])
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
print("Plot: \n")
plt.plot(x_scaled, y_scaled)
plt.xlabel("distance")
plt.ylabel("Speed")
plt.show()


# %%

# extract the times from beh_data where phase is init to reward true, find the the corresponding coordinates in the

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

# %%


init_counter = 0
lz_counter = 0
reward_time_list = []
trial_list = []

trial = init_rew_beh.iloc[index, 1]
trial_list += [trial]
init_reward_time_list = []
lz_reward_time_list = []
end_time = 0
reward_list = []

for index, row in init_rew_beh.iterrows():

    if init_rew_beh.iloc[index, 3] == 1:

        # find_start_end(t0, index, end_time)
        reward = init_rew_beh.iloc[index, 2]
        reward_list += [reward]

        init_reward_time = init_rew_beh.iloc[index, 0]
        init_reward_time_list += [init_reward_time]
        init_counter += 1

        # reward_time = init_rew_beh.iloc[index, 0]
        # reward_time_list += [reward_time]
    elif init_rew_beh.iloc[index, 4] == 1:

        # find_start_end(tN, index, end_time)

        reward = init_rew_beh.iloc[index, 2]
        reward_list += [reward]

        lz_reward_time = init_rew_beh.iloc[index, 0]
        lz_reward_time_list += [lz_reward_time]

        lz_counter += 1
    else:
        index += 1

x = np.array(reward_list)
print(np.unique(x))

unique, counts = np.unique(reward_list, return_counts=True)

result = np.column_stack((unique, counts))
print(result)


start_time = init_reward_time_list[0]
end_time = lz_reward_time_list[-1]

print("Start time is ", start_time, "and end time is ", end_time)


# a = pd.Series(np.fill(np.nan, len(df)))
# a[df.c_zone] = 1.0
# a[df.left_arm] = 2.0
# a[df.left_arm.diff()==1] = 0.0
# a = a.ffill()


# %% function


# %% merging truth values from other columns to central zone


init_rew_beh["Central_Zone"][init_rew_beh["Central_Zone"] == True] = 1
init_rew_beh["Central_Zone"][init_rew_beh["Central_Zone"] == False] = 2
init_rew_beh["L_Zone"][init_rew_beh["L_Zone"] == False] = 2
init_rew_beh["L_Zone"][init_rew_beh["L_Zone"] == True] = 3

# %%

# drop Rz
# init_rew_beh.drop('R_Zone', inplace=True, axis=1)


# replace the 2 values in C_zone with 3 if L_Zone is 3
init_rew_beh["Central_Zone"][init_rew_beh["L_Zone"] == 3] = 3


# %% extract all correct left-turn trials

warnings.simplefilter(action='ignore', category=FutureWarning)


def start_and_end_time_of_trial(start_time, end_time, trial_number):
    correct_trial_DF = pd.DataFrame([], columns=['Trial Number', 'Start time', 'End time'])
    correct_trial_DF = correct_trial_DF.append(
        {'Trial Number': trial_number, 'Start time': start_time, 'End time': end_time}, ignore_index=True)
    return correct_trial_DF


# take the time values from cz 1 -> 3
trial_DF = pd.DataFrame([], columns=['Timestamps', 'Value'])
end_time = 0
start_time = -1
correct_sequence = False
trial_number = 0
Cz2 = False
Cz3 = False
dataframe_collection = {}
trial_list = []


for index, row, in init_rew_beh.iterrows():  # for every row in df

    # if value of column Central Zone is 1 (True)
    if init_rew_beh.iloc[index, 4] == 1 and Cz2 == False:

        if start_time == -1:  # if start_time hasnt been found
            start_time = init_rew_beh.iloc[index, 0]  # start time is the time in index row

        # either way add the index row in a new df
        trial_DF = trial_DF.append(
            {'Timestamps': init_rew_beh.iloc[index, 0], 'Value': init_rew_beh.iloc[index, 4]}, ignore_index=True)
        correct_sequence = True

        # start_and_end_time_of_trial(start_time, end_time, trial_DF) # send start and end time in function

        # if init_rew_beh.iloc[index + 1, 3] == 2:
        #     trial_DF = trial_DF.append(
        #         {'Timestamps': init_rew_beh.iloc[index, 0], 'Value': init_rew_beh.iloc[index, 3]}, ignore_index=True)
        #     correct_sequence = True

    elif init_rew_beh.iloc[index, 4] == 2 and correct_sequence == True and Cz3 == False:
        trial_DF = trial_DF.append(
            {'Timestamps': init_rew_beh.iloc[index, 0], 'Value': init_rew_beh.iloc[index, 4]}, ignore_index=True)
        # correct_sequence = True
        Cz2 = True

    elif init_rew_beh.iloc[index, 4] == 3 and correct_sequence == True:
        trial_DF = trial_DF.append(
            {'Timestamps': init_rew_beh.iloc[index, 0], 'Value': init_rew_beh.iloc[index, 4]}, ignore_index=True)
        # correct_sequence = True

        if init_rew_beh.iloc[index + 1, 4] == 2:
            correct_sequence = False
            trial_number += 1
            end_time = init_rew_beh.iloc[index, 0]
            correct_trials = start_and_end_time_of_trial(start_time, end_time, trial_number)
            # print("Correct_trials\n", correct_trials)
            sleep(2)
            trial_list += [correct_trials]
            # reset variables
            Cz2 = False
            start_time = -1
            Cz3 = True

print(trial_list)


# DF= pd.DataFrame({'chr': ["chr3", "chr3", "chr7", "chr6", "chr1"], 'pos': [10, 20, 30, 40, 50], })
# ans= [y for x, y in DF.groupby('chr')]

# %% extract time range from dlc


# correct_trial_dlc = dlc_data.where(
#     ((dlc_data.iloc[:, 24] > 41.275456) & (dlc_data.iloc[:, 24] < 92.847808)) & ((dlc_data.iloc[:, 24] > 210.618432) & (dlc_data.iloc[:, 24] < 325.368512)) & ((dlc_data.iloc[:, 24] > 369.276992) & (dlc_data.iloc[:, 24] < 487.094144)) & ((dlc_data.iloc[:, 24] > 568.859405) & (dlc_data.iloc[:, 24] < 787.692006)) & ((dlc_data.iloc[:, 24] > 890.07648) & (dlc_data.iloc[:, 24] < 1081.788915))).dropna()


# %% Extract one correct trial


correct_trial_Cz = init_rew_beh.where(init_rew_beh.iloc[:, 1] == 6).dropna()
correct_trial_Lz = init_rew_beh.where(
    (init_rew_beh.iloc[:, 1] == 7) & (init_rew_beh.iloc[:, 4] == 1)).dropna()
frames = [correct_trial_Cz, correct_trial_Lz]
correct_trial = pd.concat(frames)

extract_time = correct_trial.iloc[-1, 0] - correct_trial.iloc[0, 0]
print("Start time is ", correct_trial.iloc[0, 0], " and end time is ",
      correct_trial.iloc[-1, 0], "and the difference is ", extract_time)


correct_trial_dlc = dlc_data.where(
    (dlc_data.iloc[:, 24] > 261.55) & (dlc_data.iloc[:, 24] < 360.87)).dropna()

# %% plot


# Find and plot only left turn coordinates

col0 = correct_trial_dlc.iloc[:-1, 18]  # x coordinates
col1 = correct_trial_dlc.iloc[:-1, 19]  # y coordinates
col2 = correct_trial_dlc.iloc[:-1, 24]  # time
col3 = speed_df.iloc[:-1, 0]  # speed values
coords_df = pd.concat([col0, col1, col2, col3], axis=1)  # x, y coordinates dataframe and time
# rename column names
# coords_df = coords_df.rename(columns={
#     0: 'x', 1: 'y'})

print("X coordinates: ", col0.size, "& ", col0.isnull().sum(), " missing values")
print("Y coordinates: ", col1.size, "& ", col1.isnull().sum(), " missing values")
print("Time: ", col2.size, "& ", col2.isnull().sum(), " missing values")
print("Speed values: ", col3.size, "& ", col3.isnull().sum(), " missing values")
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

coords_df = pd.concat([lcx, lcy, col2, col3], axis=1)
coords_df.isnull().sum()

coords_df = coords_df[coords_df.iloc[:, 1] != 0]


plt.scatter(coords_df.iloc[:, 0], coords_df.iloc[:, 1], s=0.01)
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


# %%

# slice a dataset in bins
set_bins = 250
coords_quartiles = np.array(coords_df)
coords_quartiles = np.array_split(coords_quartiles, set_bins)

average_speed_list = []
for index in range(set_bins):
    average_speed = np.mean(coords_quartiles[index][:, 3])
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
print("Plot: \n")
plt.plot(x_scaled, y_scaled)
plt.xlabel("distance")
plt.ylabel("Speed")
plt.show()

# con = np.concatenate((x_scaled, y_scaled), axis=1)

# %%


# %%


# for index, row in init_rew_beh.iterrows():
#     if init_rew_beh.iloc[index, 2] == 1:


#         reward_time = init_rew_beh.iloc[index, 0]
#         reward_time_list += [reward_time]
#     elif init_rew_beh.iloc[index, 3] == 1:


#         reward_time = init_rew_beh.iloc[index, 0]
#         reward_time_list += [reward_time]
#     else:
#         index += 1


final_reward_time = reward_time_list[-1] - reward_time_list[0]


# for index, row in dlc_data.iterrows():

#     # control if to exit the function
#     if dlc_data.iat[index, 24] == dlc_data.iat[-1, 24]:
#         print("All speed values have been stored in list successfully.")
#         break

#     x1 = dlc_data.iat[index, 18]
#     x2 = dlc_data.iat[index+1, 18]
#     y1 = dlc_data.iat[index, 19]
#     y2 = dlc_data.iat[index + 1, 19]

reward_trials = init_rew_beh.where(
    (init_rew_beh.iloc[:, 1] == 'True') | (init_rew_beh.iloc[:, 2] == 'True'))


init_rew_beh["Central_Zone"] = init_rew_beh["Central_Zone"].astype("category")
# cast categorical values to numercical, 0 = False, 1 = Incorrect


# %% total distance of left turn
# get the start time
st = time.time()

# List where speed values will be stored
left_turn_speeds_list = []
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

    left_turn_current_distance = calculate_distance.dist_calc(x1, x2, y1, y2)
    left_current_speed = left_turn_current_distance / \
        (coords_df.iat[index+1, 2]-coords_df.iat[index, 2])

    total_left_turn_distance = total_left_turn_distance + left_turn_current_distance

    total_left_turn_distance_list += [total_left_turn_distance]

    left_turn_distance_list += [left_turn_current_distance]
    # print(left_turn_distance_list[index])
    left_turn_speeds_list += [left_current_speed]

print(len(left_turn_distance_list))  # list with all the distances
print(len(left_turn_speeds_list))  # list with all the different speeds
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

# plt.plot(total_left_turn_distance_list, left_turn_speeds_list)
# plt.xlabel("distance")
# plt.ylabel("Speed")
# plt.show()


# %%

plt.plot(np.arange(total_left_turn_distance), average_speed_list)
plt.xlabel("bins")
plt.ylabel("avg speeds")
plt.show()
# ============


# calculate distnace only for left-turn coordinates


plt.plot(np.arange(total_left_turn_distance), left_turn_speeds_list)
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
