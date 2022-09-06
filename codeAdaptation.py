#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 13:15:00 2022

@author: konstantinos kalaitzidis
"""
#%% Importing Packages and Libraries

#numeric analysis
import pandas as pd
import numpy as np
from datetime import datetime
import math

# Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns

#%% Functions

#function to calculate distance between the coordinates
def calculateDistance(x1, x2, y1, y2):
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

 

#print information for the data files
def datasetInfo():
    
        def dataNames(int):
            names = ["Behavioral", "Tracking", "h5"]
            return print(names[i], "INFORMATION:")
                
    
        datafile_names = {'behavioral data' : beh_data, 'tracking data': dlc_data, 'h5':f}
        i = 0
        for key in datafile_names:
            dataNames(i)
            print(datafile_names[key].info(), "\n")
            i = i + 1
    
#determine if your dataset has missing values 
def missingInfo():
    print("Behavioral:\n",beh_data.isnull().sum(), "\n")
    print("Tracking:\n",dlc_data.isnull().sum(), "\n")
    print("Calcium:\n",f.isnull().sum(), "\n")
    sns.heatmap(beh_data.isnull())
    sns.heatmap(dlc_data.isnull())
    sns.heatmap(f.isnull())


#see where we have the firt ca occurnace and print it's timestamp
def linear_search(values, search_for):
    
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
          
          
          
          
        

#%% Pre-processing

#preparing behavioral data file
beh_data = pd.read_csv("/Users/pierre.le.merre/OneDrive - KI.SE/Mac/Desktop/KIlab/data files/Behavioral Region of Interest/tmaze_2021-10-16T17_05_25.csv")

#rename column '2021-10-16T17:05:26.6032384+02:00' to 'Time'
beh_data = beh_data.rename(columns={'2021-10-16T17:05:26.6032384+02:00':'Time', '0':'Choice', '0.1':'Init-Reward', 'False':'Initiation', 'False.1':'Incorrect', 'False.2':'Reward'})
#calcium_detection_times = calcium_detection_times.rename(columns={'Incorrect_ROI': 'Initiation_ROI'})



#Preparing deep lab cut file
dlc_data = pd.read_hdf("/Users/pierre.le.merre/OneDrive - KI.SE/Mac/Desktop/KIlab/data files/Deep lab cut/tmaze_2021-10-16T16_39_15DLC_resnet50_Oprm1_insc_VAS_reversalNov18shuffle1_1030000.h5")

#add a column named "Time", and fill it with 0 values. 
dlc_data[dlc_data.shape[1]] = 0
dlc_data = dlc_data.rename(columns={24: 'Time'})



#reading the arrowmaze_data2.h5 file^
#write code to call the functions from the readSessions.py file. 
import pandas as pd
from utils import readSessions


# =============================================================================
# #Print the shape of the calcium imaging and deeplabcut tracking for all sessions
# for session in readSessions.findSessions("/Users/pierre.le.merre/OneDrive - KI.SE/Mac/Desktop/arrowmaze_project-main/arrowmaze_data2.h5"):
#     S = session.readDeconvolvedTraces()
#     tracking = session.readTracking()
#     print(S.shape, tracking.shape if tracking is not None else None)
#     
# #Loop through only sessions of animal 2 and print the timestamp of the deeplabcut video
# for session in readSessions.findSessions("/Users/pierre.le.merre/OneDrive - KI.SE/Mac/Desktop/arrowmaze_project-main/arrowmaze_data2.h5", animal_no=2):#
#     print(session.meta.video_time)
# =============================================================================


pathname = "/Users/pierre.le.merre/OneDrive - KI.SE/Mac/Desktop/arrowmaze_project-main/striatum-2choice/data/arrowmaze_data2.h5"
#reading the h5 file that contains the deeplabcut and calcium imaging
with pd.HDFStore(pathname) as hdf:
    # This prints a list of all group names:
    print(hdf.keys())

f = pd.read_hdf(pathname, key="/meta")




# =============================================================================
# now you can open whichever subfile you want, for example: 
# f1 = pd.read_hdf("/Users/pierre.le.merre/OneDrive - KI.SE/Mac/Desktop/arrowmaze_project-main/arrowmaze_data2.h5", key="")
# 
# 
# g = pd.read_hdf("/Users/pierre.le.merre/OneDrive - KI.SE/Mac/Desktop/arrowmaze_project-main/arrowmaze_data2.h5", key="/ca_recordings/20211016_163921_animal1learnday1/S")
# 
# 
# df_animal_no2 = pd.read_hdf("/Users/pierre.le.merre/OneDrive - KI.SE/Mac/Desktop/arrowmaze_project-main/arrowmaze_data2.h5")
# df_animal_no2.head()
# =============================================================================


'''Removing the date part from the datetime values to keep only seconds'''
#converting the 'Time' column to time format
beh_data['Time'] =  pd.to_datetime(beh_data['Time'])  


#subtracting the first time element from all the other times in the column
beh_data.iloc[:,0] = beh_data.iloc[:,0] - beh_data.iloc[0, 0]


#convert column to seconds
beh_data['Time'] = beh_data.iloc[:,0].dt.total_seconds()


'''Adding the time to the dlc_data file with the according step.'''

#Finding the step: total time (last time element) / the number of rows of the dlc file
total_time = beh_data.iat[len(beh_data) - 1, 0]   # 1233.927104
step = total_time / len(dlc_data)
#print(step)                                      #0.016609375348292526


#fill in data in the last column (Time column) with the time
dlc_data.iloc[:, 24] = range(len(dlc_data))*step


#%% Understanding our datasets


#What are some information about our datasets? 
#datasetInfo()

    
# Checking for missing data
#missingInfo()

#%% Store processed data

#%% Calculations


'''Calculate the average speed, the standard deviation, and the standard error of the mean 
of the mouse on the behavioral file according to the coordinates on the dlc file'''

speed_list = [] #list where speed values will be stored


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
    
    distance = calculateDistance(x1, x2, y1, y2)
    speed = distance / (dlc_data.iat[index+1, 24]-dlc_data.iat[index, 24])
    
    speed_list += [speed]
    
    
#average speed
print("The mouse's average speed is: ", np.mean(speed_list))

#standard deviation
print("The standard deviation is: ", np.std(speed_list))

#standard error of the mean
print("The standard error of the mean is: ", np.std(speed_list) / np.sqrt(np.size(speed_list)))


#sort array in ascending order
sorted_speed_list = np.sort(speed_list)


#find the max speed first.
max_speed = np.max(sorted_speed_list) #48297.263931042384
min_speed = np.min(sorted_speed_list) #0.20031764027047433


#%% Gaussian filtering

'''Perform guassian filtering in the speed list'''

from scipy.stats import gaussian_kde
speed_list=np.array(speed_list)

# stacking the X and Y coordinates columns vertically
xy=np.vstack([dlc_data.iloc[:, 18],range(len( dlc_data.iloc[:, 19]))])

#applying gaussian filtering
z=gaussian_kde(xy)(xy)
fig1,ax=plt.subplots(1,1)

#plot
ax.scatter(dlc_data.iloc[:, 18],dlc_data.iloc[:, 19],c=z,s=1)
plt.show()


#%% Creating quartiles

'''Split the dlc_data array into 4 equal parts (quartiles)'''

#specify how many parts
set_quartiles = 4
                                     
quartiles = np.array(dlc_data)
quartiles = np.array_split(quartiles, set_quartiles)

#%%  Plotting


'''Create a plot with N number of subplots showing the different quartiles of
of the dlc_data and apply guassia smoothing to each subplot'''


plotRows = 2 
plotColumns = 2

# Creating 4 (or N) subplots and unpacking the output array immediately


#Perform guassian filtering in the data  
from scipy.stats import gaussian_kde


fig2, axs=plt.subplots(plotRows, plotColumns)


axes_list=[axs[0,0],axs[0,1],axs[1,0],axs[1,1]]


for ax,i in zip(axes_list,range(set_quartiles)):
    
    #filtering for each quartile
    xy=np.vstack([quartiles[i][:, 18], range(len(quartiles[i][:, 19]))])
    z=gaussian_kde(xy)(xy)

    #plotting for each quartile
    ax.scatter(quartiles[i][:, 18], quartiles[i][:, 19], c=z, s=1)

plt.tight_layout()



#%% Find the timestamps of ca detection


calcium_detection_times = beh_data.drop_duplicates(subset=['0.3'])


# see where we have the firt ca occurnace and print it's timestamp
detect_time = linear_search(beh_data.iloc[:, 7], 1)

# =============================================================================
# datafile_name = dlc_data
# detect_time = linear_search(datafile_name.iloc[:, 24], 1)
# =============================================================================

#subtracting the first time of ca detection from all the previous times
calcium_detection_times.iloc[:, 0] = calcium_detection_times.iloc[:, 0] - detect_time
calcium_detection_times.drop_duplicates(subset=None, keep='first', inplace=True, ignore_index=True)



#%% append ca times to dlc_data df


dlc_data.iloc[:, 24] = dlc_data.iloc[:, 24] - detect_time



#%% Phase detection 

# prep the df
# change the names

calcium_detection_times[calcium_detection_times.shape[1]] = 0
calcium_detection_times.rename(columns={"8":"ROI"}, inplace=True)

calcium=calcium_detection_times
calcium[8][calcium['Initiation']==True]='Initiation'
calcium[8][calcium['Incorrect']==True]='Incorrect'
calcium[8][calcium['Reward']==True]='Reward'
calcium[8][(calcium['Initiation']==False) & (calcium['Reward']==False) ]='Task'


#%% Modifying the ca file



#Print the shape of the calcium imaging and deeplabcut tracking for all sessions
# =============================================================================
# for session in readSessions.findSessions("data/arrowmaze_data.h5"):
#     S = session.readDeconvolvedTraces()
#     tracking = session.readTracking()
#     print(S.shape, tracking.shape if tracking is not None else None)
# =============================================================================
# =============================================================================
#     
# #Loop through only sessions of animal 2 and print the timestamp of the deeplabcut video
# for session in readSessions.findSessions("data/arrowmaze_data.h5", animal_no=2):
#     print(session.meta.video_time)
# 
# 
# 
# def readCalciumTraces(self):
#         '''Read all calcium traces from this session.
#         Returns:
#         A Pandas dataframe with the calcium traces as columns
#         '''
#         path = "/ca_recordings/{}/C".format(self.meta.ca_recording)
#         return pd.read_hdf(self.hdfFile, path).unstack(level=0).C
# 
# 
# 
# def findSessions(hdfFile, filterQuery=None, sortBy=None, closeStore=True, **filters):
#     store = pd.HDFStore(hdfFile, 'r')
#     queries = []
#     for col, val in filters.items():
#         if isinstance(val, str):
#             queries.append("{} == '{}'".format(col, val))
#         elif isinstance(val, list) or isinstance(val, tuple):
#             queries.append("{} in {}".format(col, val))
#         elif isinstance(val, int):
#             queries.append("{} == {}".format(col, val))
#         else:
#             raise ValueError("Unknown filter type")
#     meta = pd.read_hdf(store, "/meta")
#     if filterQuery is not None: meta = meta.query(filterQuery)
#     if queries: meta = meta.query(" & ".join(queries))
#     if sortBy is not None: meta = meta.sort_values(sortBy)
#     for sessionMeta in meta.itertuples():
#         yield Session(store, sessionMeta)
#     if closeStore: store.close()
# 
# =============================================================================




#%% Reading .nwb file

# =============================================================================
# import numpy as np
# from pynwb import NWBHDF5IO
# 
# io = NWBHDF5IO('/Users/pierre.le.merre/OneDrive - KI.SE/Mac/Desktop/arrowmaze_project-main/lab-content/20211016_163921_animal1learnday1.nwb', 'r')
# nwbfile_in = io.read()
# 
# 
# 
# 
# 
# from pynwb import NWBFile, NWBHDF5IO, TimeSeries
# import datetime
# import numpy as np
# 
# # first, write a test NWB file
# nwbfile = NWBFile(
#     session_description='demonstrate adding to an NWB file',
#     identifier='NWB123',
#     session_start_time=datetime.datetime.now(datetime.timezone.utc),
# )
# 
# filename = '/Users/pierre.le.merre/OneDrive - KI.SE/Mac/Desktop/arrowmaze_project-main/lab-content/20211016_163921_animal1learnday1.nwb'
# # =============================================================================
# # with NWBHDF5IO(filename, 'w') as io:
# #     io.write(nwbfile)
# # =============================================================================
# 
# # open the NWB file in r+ mode
# with NWBHDF5IO(filename, 'r+') as io:
#     read_nwbfile = io.read()
# 
# 
# =============================================================================

#%% Notes

# =============================================================================


'''Optimized way to plot'''
# =============================================================================
# for ax,i in zip(axes_list,range(set_quartiles)):
#     ax.scatter(quartiles[i][:, 18], quartiles[i][:, 19] )
# plt.tight_layout()
# =============================================================================



'''My solution to plot the subplot'''

# =============================================================================
# j=0    
# for i in range(set_quartiles):
#     if i >= plotRows:  
#         if j == 0:
#             axs[j,j+1].scatter(quartiles[i][:, 18], quartiles[i][:, 19]) #c=z,s=1)
#             xy=np.vstack([quartiles[i][:, 18], range(len(quartiles[i][:, 19]))])
#             z=gaussian_kde(xy)(xy)
#             j = j + 1
#         elif j == 1:
#             axs[j,j].scatter(quartiles[i][:, 18], quartiles[i][:, 19]) #c=z,s=1)
#             xy=np.vstack([quartiles[i][:, 18], range(len(quartiles[i][:, 19]))])
#             z=gaussian_kde(xy)(xy)
#     if i < plotRows:      
#         axs[i,0].scatter(quartiles[i][:, 18], quartiles[i][:, 19]) #c=z,s=1) 
#         xy=np.vstack([quartiles[i][:, 18], range(len(quartiles[i][:, 19]))])
#         z=gaussian_kde(xy)(xy)
# 
# 
# plt.tight_layout()
# =============================================================================



'''My solution to plot the subplot'''

# =============================================================================
# fig2, axs=plt.subplots(plotRows, plotColumns)
# 
# j=0    
# for i in range(set_quartiles):
#     if i >= plotRows:  
#         if j == 0:
#             axs[j,j+1].scatter(quartiles[i][:, 18], quartiles[i][:, 19]) #c=z,s=1)
#             j = j + 1
#         elif j == 1:
#             axs[j,j].scatter(quartiles[i][:, 18], quartiles[i][:, 19]) #c=z,s=1)
# 
#     if i < plotRows:      
#         axs[i,0].scatter(quartiles[i][:, 18], quartiles[i][:, 19]) #c=z,s=1) 
#     
# 
# 
# plt.tight_layout()
# =============================================================================





'''Code snippets'''


#plotting subplots "manually"

# =============================================================================
# fig, axs = plt.subplots(2, 2)
# fig.suptitle('DLC_data Quartiles')
# 
# axs[0, 0].scatter(x0, y0, c='blue')
# axs[0, 0].set_title('Quartile 1')
# 
# axs[0, 1].scatter(x1, y1, c='orange')
# axs[0, 1].set_title('Quartile 2')
# 
# 
# axs[1, 0].scatter(x2, y2, c='green')
# axs[1, 0].set_title('Quartile 3')
# 
# axs[1, 1].scatter(x3, y3, c='red')
# axs[1, 1].set_title('Quartile 4')
# 
# plt.tight_layout()
# =============================================================================




#Split the data of the file into 4 parts (or quartiles)
# =============================================================================
# quartiles = np.array(dlc_data)
# quartiles = np.array_split(quartiles, 4)
# =============================================================================
















