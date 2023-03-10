# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 09:47:01 2021

@author: TomsS


Needs rewrite to allow for chanels that have not been named in UDL
"""
import pandas as pd
import numpy as np


def find_data_start(fid):
    """ Returns first data line """
    line = fid.readline()
    split_line = line.strip().split(',')
    while split_line[0].strip() != "Channel >>": #Find the channel line and base off that
        split_line = line.split(',')
        line = fid.readline()
    return line


def find_channels(line):
    """ Finds the chanels, and returns the number of index columns """
    channels = line.strip().split(',')
    num_index_cols = 1 #Start with at least one index column (UDL requires this)
    while channels[num_index_cols] == "": #Check if date and time fields are used
        num_index_cols += 1
        
    channel_names = []
    channel_numbers = []
    for channel in channels[num_index_cols:-1]:
        i = find_bracket(channel)
        if i == 0:
            channel_number = channel.strip()
            channel_name = channel_number
        else:
            channel_name =  channel[:i].strip()
            channel_number = channel[i+1:-1]
        channel_names.append(channel_name)
        channel_numbers.append(channel_number)
    channel_names = pd.Series(channel_names, index=channel_numbers)
    return channel_names, channel_numbers, num_index_cols


def find_bracket(string):
    """ Finds the last occurance of ( in a string and returns -ve index"""
    i = -1
    while string[i] != '(':
        if abs(i) == len(string):
            return 0
        i -= 1   
    return i
    

def find_remaining_properties(fid, num_index_cols, channel_numbers):
    """ Finds the remaining properties, multiplier, constants, units """
    line = fid.readline()
    multiplier = pd.Series([float(i) for i in line.strip().split(',')[num_index_cols:-1]], index=channel_numbers)  #Drop off the trailing newline character
    line = fid.readline()
    constants = pd.Series([float(i) for i in line.strip().split(',')[num_index_cols:-1]], index=channel_numbers) #Drop off the trailing newline character
    line = fid.readline()
    units = pd.Series(line.strip().split(',')[num_index_cols:-1], index=channel_numbers)
    index_header = line.strip().split(',')[:num_index_cols]
    return multiplier, constants, units, index_header


class UDL_file:
    
    def __init__(self, filename):
        
        fid = open(filename, 'r')
        line = find_data_start(fid)
         
        
        #Should be at the chanel line here.
        self.channel_names, self.channel_numbers, num_index_cols = find_channels(line)
        
        self.multipliers, self.constants, self.units, index_header = find_remaining_properties(fid, num_index_cols, self.channel_numbers)
        
        self.header = index_header + self.channel_numbers
        self.data = pd.read_csv(fid, usecols=range(0,len(self.header)), names=self.header)
        fid.close()
        
    def change_multiplier(self, new_multipliers, verbose=False):
        """ Takes new multipliers """
        for number, multiplier in new_multipliers:
            try:
                data = self.data[number]
                data = np.around(data / self.multipliers[number]) * multiplier
                constant = np.around(self.constants[number] / self.multipliers[number]) * multiplier
                difference = (multiplier - self.multipliers[number])/self.multipliers[number] * 100
                if verbose:
                    print(f"Channel {self.channel_names[number]} ({number}) was changed from {self.multipliers[number]:0.4f} to {multiplier:0.4f} with {difference:0.2f}% difference")
                self.data[number] = data
                self.multipliers[number] = multiplier
                self.constants[number] = constant
            except:
                print(f"ERROR: changing {number} to {multiplier} failed")
                
    def only_trigger_points(self, trigger_col_name, threshold=1000):
        
        trigger = self.data[trigger_col_name]
        
        selection_set = np.concatenate(([True], abs(np.diff(trigger)) > threshold))
        self.old_data = self.data
        self.data = self.data[selection_set]
        
        
        
            

    
#file_directory = r'C:\Users\tdw42\OneDrive - University of Canterbury\PhD\Testing\Wing Lab\6W 12S Monotonic'
#filename = r"C:\Users\tdw42\OneDrive - University of Canterbury\PhD\Testing\Wing Lab\Tests\12W 24S Monotonic REPAIRED\2020-11-17 - 12W 24S REPAIRED Monotonic #1 #11.5 UDL.csv"

#file = UDL_file(filename)
#file.change_multiplier([('A2', 0.077990306819829), ('B1', -0.025095569975106), ('B2', -0.025506555062862), ('B3', 0.051125230300698), ('B4', -0.007657348138896), ('B5', 0.088761306634472), ('B6', 0.007593299144795), ('B7', -0.007563385255353)])

        
        
        
            