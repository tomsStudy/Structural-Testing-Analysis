# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 12:40:45 2020

@author: tdw42
"""

import os
import Test_Set
import pandas as pd

def get_folder_paths(path='.'):
    """ Gets the foldernames in the given directory """
    #filepath = 'C:\Users\tdw42\OneDrive - University of Canterbury\2019 BEGINNINGS\Testing\Model Structures Lab\2020-03-09\'
    entries = os.scandir(path)
    print("Folders in Current Directory:")
    csv_filenames = []
    for entry in entries:
        if entry.is_dir(): #Could change this to a regular expression
            csv_filenames.append(entry.name)
            print("    ", entry.name)
        #end
    #end
    return csv_filenames
#end


def run_folder(path, function):
    folder_names = get_folder_paths(path)
    i = 0
    for folder_name in folder_names:
        #Run connection script Pass folder name and path+name + funciton
        print(f"Running folder: {folder_name}")
        
        test_set = Test_Set.Test_Set(folder_name, path, function)
        test_set.import_tests()
        test_set.export_summary()
        test_set.export_curves()
        test_set.export_plots()
        
        i += 1
    print(f"Complete: {i} folders run")
    
    
def consolodate_summary_sheets(path):
    
    df = pd.DataFrame()
    folder_names = get_folder_paths(path)
    
   
    try:
        split_path = path.split("\\")
        sheet_name = f'{split_path[-2]} Results Summary'
        
        writer = pd.ExcelWriter(path+"\\"+ sheet_name+'.xlsx', engine='xlsxwriter')
        #writer.book.add_worksheet(sheet_name)
        if len(sheet_name) > 31:
            sheet_name = sheet_name[:31]
        df = pd.DataFrame()
        df.to_excel(writer, index=False, sheet_name=sheet_name) #Have to do this otherwise xlsxwriter locks out pandas # , float_format = "%0.4g"
        sheet = writer.sheets[sheet_name]
        start_row = 0
        for folder_name in folder_names:
            print(f"Collating folder: {folder_name}")
            filenames = get_summary_filenames(path, folder_name)
            for file in filenames:
                filename = path + r"\\" + folder_name + r"\\" + file
                df = pd.read_excel(filename)
                sheet.write(start_row, 0, folder_name)
                start_row += 1
                
                df.to_excel(writer, index=False, sheet_name=sheet_name, startrow=start_row)  #, float_format = "%0.4g"
                start_row += df.shape[0] + 2 
                
                
                
                
                
        writer.close()
        
        
        
    except Exception as e:
        print("ERROR: Writing summary file {}")
        print(e)
    
    
            
            
            
def get_summary_filenames(path, folder_name):
    """ Gets the CSV filenames in the given directory """
    path = path + '\\' + folder_name
    entries = os.scandir(path)
    print("CSV Files in Current Directory:")
    filenames = []
    for entry in entries:
        if entry.name == folder_name + '.xlsx': #This is looking for the summary file
            filenames.append(entry.name)
            print("    ", entry.name)
        #end
    #end
    return filenames
#end
#end
    
    
    
    