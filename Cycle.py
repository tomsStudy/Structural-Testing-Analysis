# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 15:28:01 2021

@author: tdw42
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import UDL_import as UDL 
# from Test import *

# """
# def split_cycles(dataframe, threshold=0.02, min_pts_between=5):
#     disp = dataframe['control']
#     condition = disp<threshold
#     indices = np.arange(0, len(disp))[condition]
#     index_arrays = np.split(indices, np.where(np.diff(indices) > min_pts_between)[0]+1)
#     zero_points = []
#     for index_array in index_arrays:
#         if len(index_array) > 0:
#             min_index_index = np.argmin(disp[index_array])
#             min_index = index_array[min_index_index]
#             zero_points.append(min_index)
    
#     if len(zero_points) > 0:
#         if zero_points[0] != 0:
#             zero_points = [0] + zero_points
#     else:
#         zero_points = [0] + zero_points
#     print(zero_points)
#     if zero_points[-1] != len(disp) -1:
#         zero_points.append(len(disp) -1) #Add on a final split point for monotonic cycles
    
#     #Now split into cycles between 0 points
#     cycles = []
#     for i in range(1, len(zero_points)):
#         cycles.append(Cycle(dataframe.iloc[zero_points[i-1]:zero_points[i]+1], i))
#     return cycles

# def smooth_cycles(cycles, y_col):
#     """ """
#     for cycle in cycles:
#         cycle.smooth_cycle1()
#         cycle.smooth_cycle2(y_col)
# """


def split_cycles(test, control_column='control', threshold=0.02, min_pts_between=5):
    disp = test.data[control_column]
    condition = disp<threshold
    indices = np.arange(0, len(disp))[condition]
    index_arrays = np.split(indices, np.where(np.diff(indices) > min_pts_between)[0]+1)
    zero_points = []
    for index_array in index_arrays:
        if len(index_array) > 0:
            min_index_index = np.argmin(disp[index_array])
            min_index = index_array[min_index_index]
            zero_points.append(min_index)
    
    if len(zero_points) > 0:
        if zero_points[0] != 0:
            zero_points = [0] + zero_points
    else:
        zero_points = [0] + zero_points
    #print(zero_points)
    if zero_points[-1] != len(disp) -1:
        zero_points.append(len(disp) -1) #Add on a final split point for monotonic cycles
    
    #Now split into cycles between 0 points
    cycles = []
    for i in range(1, len(zero_points)):
        cycles.append(Cycle(test.data.iloc[zero_points[i-1]:zero_points[i]+1], i))
    test.cycles = cycles



def smooth_cycles(test):
    """ """
    for cycle in test.cycles:
        cycle.smooth_cycle1()
        cycle.smooth_cycle2()

def reconstruct_cycles(test):
    """ """
    keep_indices = []
    for cycle in test.cycles:
        keep_indices += cycle.data.index.tolist()
    keep_indices = sorted(list(set(keep_indices))) #remove duplicates
    test.data = test.data.loc[keep_indices]


class Cycle:
    """ Class to hold all values relating to a particular cycle """
    
    def __init__(self, dataframe, cycle_number):
        self.data = dataframe
        self.cycle_number = cycle_number
        self.find_peak()
        
    def __repr__(self):
        """ Print out the following:
            - Cycle number
            - Number of points in cycle
            - Displacement of control pot at peak"""
        return f"Cycle number: {self.cycle_number}\nCycle points: {len(self.data)}\nPeak Control: {self.peak['control']:0.2f} mm\n"
        
    def find_peak(self, x_col='disp'):
        """ Finds the peak of the cycle and evaulates the following variables:
            - peak - series of all columns at peak datapoint
            - forward - dataframe of all rows before peak
            - back - dataframe of all rows including and after peak
        """
        self.peak_i = np.argmax(abs(self.data[x_col]))
        self.peak = self.data.iloc[self.peak_i]
        self.forward = self.data.iloc[:self.peak_i+1] #Does include the peak row
        self.back = self.data.iloc[self.peak_i:]
        
    def max_force(self):
        force = self.data['force']
        disp = self.data['disp']
        max_i = np.argmax(force)
        
        max_force = force.iloc[max_i]
        max_force_disp = disp.iloc[max_i]
        return max_force, max_force_disp
    
    def smooth_cycle1(self, x_col='disp'):
        """ Removes any duplicates in data by making a set """
        self.forward = remove_not_increasing(self.forward, x_col)
        self.back = remove_not_increasing(self.back, x_col)
        keep_indices = []
        keep_indices += self.forward.index.tolist()
        keep_indices += self.back.index.tolist()
        keep_indices = sorted(list(set(keep_indices))) #remove duplicates
        self.data = self.data.loc[keep_indices]
        
    def smooth_cycle2(self, y_col='force', x_col='disp', grad_threshold=100):
        """ Uses remove spikes function to trim out unwanted jumps in data"""
        self.forward = remove_spikes(self.forward, x_col, y_col)
        self.back = remove_spikes(self.back, x_col, y_col, grad_threshold)
        keep_indices = []
        keep_indices += self.forward.index.tolist()
        keep_indices += self.back.index.tolist()
        keep_indices = sorted(list(set(keep_indices))) #remove duplicates
        self.data = self.data.loc[keep_indices]
        
    def plot(self):
        fig, ax = plt.subplots()
        ax.plot(self.data['disp'], self.data['force'])
       # plt.show()
    
def remove_not_increasing(dataframe, x_col):
    disp = dataframe[x_col]
    data_grad = disp.iloc[-1] - disp.iloc[0]
    keep_list = []
    cur_max = disp.iloc[0]
    keep_list.append(0)
    if data_grad >= 0:
        for i, value in enumerate(disp):
            if value > cur_max:
                keep_list.append(i)
                cur_max = value
    else:
        for i, value in enumerate(disp):
            if value < cur_max:
                keep_list.append(i)
                cur_max = value
    return dataframe.iloc[keep_list]
    


def remove_spikes(dataframe, x_col, y_col, grad_threshold=100):
    dataframe = dataframe.iloc[::-1]
    disp = dataframe[x_col]
    force = dataframe[y_col]
    
    i = 1
    while i < len(force)-1:
        grad_before = (force.iloc[i-1] - force.iloc[i]) / (disp.iloc[i-1] - disp.iloc[i])
        # print(disp)
        # print('force', force.iloc[i])
        # print('disp', disp.iloc[i])
        # print('grad_before', grad_before)
        
        grad_after = (force.iloc[i] - force.iloc[i+1]) / (disp.iloc[i] - disp.iloc[i+1])
        # print('grad_after', grad_after)
        if np.sign(grad_before) == 1 and  np.sign(grad_after) < 1: #If gradient before is positive and after is negative. This finds fall spikes on forward and rise spikes on back.
            if (abs(grad_before) + abs(grad_after)) > grad_threshold:
                # print(dataframe.iloc[i].name)
                # print(i)
                dataframe = dataframe.drop(dataframe.iloc[i].name)
                #np.delete(short_force,i)
                #np.delete(short_disp,i)
                
                i = max(1, i - 20) #Loop back and check if there are anymore to get rid of #Note changed on 22-11-06 from 1 to 0 as incremented after
            
        i += 1
        disp = dataframe[x_col]
        force = dataframe[y_col]
    return dataframe



# #Testing
# filename = r'C:\Users\tdw42\OneDrive - University of Canterbury\PhD\Testing\Wing Lab\Tests\12W 24S Cyclic\2020-11-25 - 12W 24S Cyclic #3 #14 UDL.csv'
# #filename = r'C:\Users\tdw42\OneDrive - University of Canterbury\PhD\Testing\Wing Lab\Tests\12W 24S Monotonic\2020-11-12 - 12W 24S Monotonic #1 #11 UDL.csv'



# file = UDL.UDL_file(filename)
# #file.change_multiplier([('A2', 0.077990306819829), ('B1', -0.025095569975106), ('B2', -0.025506555062862), ('B3', 0.051125230300698), ('B4', -0.007657348138896), ('B5', 0.088761306634472), ('B6', 0.007593299144795), ('B7', -0.007563385255353)])
# print(len(file.data))
# fig1, ax1 = plt.subplots()
# ax1.plot(file.data['A2'], file.data['A1'])

# file.data.insert(0, 'force', file.data['A1'])
# file.data.insert(1, 'disp', file.data['A2'])
# file.data.insert(2, 'control', file.data['A2'])
# test = Timber_Connection_Test(file.data, filename[0:-3])
# split_cycles(test)
# smooth_cycles(test)
# reconstruct_cycles(test)

# fig2, ax2 = plt.subplots()
# ax2.plot(test.data['A2'], test.data['A1'])
# plt.show()
# print(len(test.data))
                