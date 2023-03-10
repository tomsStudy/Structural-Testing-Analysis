# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 16:41:46 2021

@author: tdw42
"""

import UDL_import as UDL 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
#from Test import Test

class Test_Set:
    
    def __init__(self, folder_name, base_path, function):
        self.folder_name = folder_name
        self.base_path = base_path
        self.function = function
        os.chdir(base_path + '\\'+ folder_name) #This might be less robust than just using the full filename always
        self.csv_filenames = self.get_csv_filenames()
        
        
    
    def export_curves(self):
        try:
            os.mkdir('curves')
        except Exception:
            pass      
        for test in self.tests:
            for curve in test.curves: # Note curve will be the dict key
                try:
                    curve_filename = f"curves\\{test.name}_{curve}.csv"
                    test.curves[curve].to_csv(curve_filename)
                except Exception as e:
                    print(f"ERROR: Exporting {curve} curve failed on {test.name}")
                    print(e)
            
            
            """
            #Save backbone data to file
            backbone_filename = 'backbone\\' + test.name + '_backbone.csv'
            curve_filename = 'backbone\\' + test.name + '_curve.csv'
            try:
                np.savetxt(backbone_filename, np.column_stack((test.force_backbone, test.disp_backbone)), delimiter=', ', header='force backbone, displacment backbone')
                np.savetxt(curve_filename, np.column_stack((test.force, test.disp)), delimiter=', ', header ='force, displacment')
            except Exception as e:
                print(f"ERROR: Exporting curve failed on {test.name}")
                print(e)"""
        

    def export_plots(self):
        try:
            os.mkdir('graphs')
        except Exception:
            pass
        try:
            
            for test in self.tests:
                for plot in test.plots: #Remember plots are stored in a dict
                    plot_filename = f"graphs\\{test.name}_{plot}.png"
                    test.plots[plot][0].savefig(plot_filename, dpi=300)
                    
        except Exception as e:
            print(f"ERROR: Failed to make {plot} plot on {test.name}")
            print(e)
    
    
    def export_summary(self):
        
        df = pd.DataFrame()   
        for test in self.tests:
            
            df = pd.concat([df, pd.DataFrame(test.values, index=[test.name])]) #, ignore_index=True)
            #df = df.append(test.values, ignore_index=True)
        #Mean
        mean = df.mean(axis=0)
        mean.name = 'Mean'
        
        #std
        std = df.std(axis=0)
        std.name = 'std'
        
        #5th percentile
        percentile_5th = df.quantile(q=0.05, axis=0)
        percentile_5th.name = '5th Percentile'
        
        percentile_95th = df.quantile(q=0.95, axis=0)
        percentile_95th.name = '95th Percentile'
        
        
        #Write values to df
        df1 = pd.DataFrame([mean, std, percentile_5th,percentile_95th]) 
        df = pd.concat([df, df1])
        # df = df.append(mean)
        # df = df.append(std)
        # df = df.append(percentile_5th)
        # df = df.append(percentile_95th)
        try:
            writer_orig = pd.ExcelWriter(self.folder_name+'.xlsx', engine='xlsxwriter')
            sheet_name = self.folder_name
            if len(self.folder_name) > 30:
                sheet_name = self.folder_name[:30]
            
            df.to_excel(writer_orig, index=True, sheet_name=sheet_name) #float_format = "%0.3g"
            
            
            writer_orig.close()
        except Exception as e:
            print("ERROR: Writing summary file {}")
            print(e)
        
        
        
        
    def import_tests(self):
        
        self.tests = []
        for filename in self.csv_filenames:
            try:
                test = self.function(filename)
                self.tests.append(test)
            except Exception as e:
                print(f"ERROR: import test on {filename}")
                print(e)
                
        
    
    def get_csv_filenames(self):
        """ Gets the CSV filenames in the given directory """
        path = self.base_path + '\\' + self.folder_name
        #filepath = 'C:\Users\tdw42\OneDrive - University of Canterbury\2019 BEGINNINGS\Testing\Model Structures Lab\2020-03-09\'
        entries = os.scandir(path)
        print("CSV Files in Current Directory:")
        csv_filenames = []
        for entry in entries:
            if entry.name[-4:] == '.csv': #Could change this to a regular expression
                csv_filenames.append(entry.name)
                print("    ", entry.name)
            #end
        #end
        return csv_filenames
    #end
#end



def interpret_medium_scale_connection_data(file):
    force = file.data['A1']
    disp4 = file.data['A2']
    disp1 = file.data['B1']
    disp2 = file.data['B2']
    disp3 = file.data['B3']
    #disp = (disp1 + disp2)/3
    disp = disp4
    force = force.to_numpy()
    disp = disp.to_numpy()
    return force, disp
