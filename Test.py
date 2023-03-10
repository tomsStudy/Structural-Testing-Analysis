# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 16:24:06 2021

@author: tdw42
"""

import Backbone
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d


class Timber_Connection_Test():
    """ COuld also load in from file and create an instance """
    def __init__(self, data, name):
        self.data = data
        self.name = name
        
        self.curves = {}
        self.values = {} #pd.Series(index=self.name) #was hard to get name to work
        self.plots = {} #pd.Series(dtype='object') #should really be a dict
    #end
        
        
    def __repr__(self):
        """ Print out quanitites could be  """
        output = f"Test Object: {self.name}\n"
        for key in self.values.keys():
            data = self.values[key]
            output += f"{key}: {data}\n"
        return output
    #end


    def force_disp(self):
        self.curves['force_disp'] = self.data[['force', 'disp']] # This is here as it otherwise misses out on any curve trimming in the cycle processing stage
    #end
        
        
    def backbone(self):
        force=self.data['force'].to_numpy()
        disp=self.data['disp'].to_numpy()        
        force_backbone, disp_backbone = Backbone.backbone(force, disp)
        # print(force_backbone) ############################# THis is still broken
        d = {'force': force_backbone, 'disp': disp_backbone} 
        # np.array([force_backbone, disp_backbone]).T, columns=['force', 'disp']
        self.curves['backbone'] = pd.DataFrame(d)
    #end
        
    
    def backbone2(self):
        force=self.data['force'].to_numpy()
        disp=self.data['disp'].to_numpy()        
        force_backbone, disp_backbone = Backbone.backbone2(force, disp)
        # print(force_backbone) ############################# THis is still broken
        d = {'force': force_backbone, 'disp': disp_backbone} 
        # np.array([force_backbone, disp_backbone]).T, columns=['force', 'disp']
        self.curves['backbone2'] = pd.DataFrame(d)
    #end
    
    
    def backbone3(self):
        force=self.data['force'].to_numpy()
        disp=self.data['disp'].to_numpy()        
        force_backbone, disp_backbone = Backbone.backbone3(self.cycles)
        # print(force_backbone) ############################# THis is still broken
        d = {'force': force_backbone, 'disp': disp_backbone} 
        # np.array([force_backbone, disp_backbone]).T, columns=['force', 'disp']
        self.curves['backbone3'] = pd.DataFrame(d)
    #end
    
    
    def elastic_stiffness_10_40(self):
        """ 
        Calculates the elastic stiffness line (grad and incercept) based on the points though 0.1F_max and 0.4F_max approach
        Written by TW 29/06/20
        Inputs:
            -force
            -displacement
        Outputs:
            -alpha_grad
            -alpha_intercept
        """ 
        if not 'backbone' in self.curves:
            self.backbone()
        
        force=self.curves['backbone']['force'].to_numpy()
        disp=self.curves['backbone']['disp'].to_numpy()
        F_max, V_F_max, F_max_i = max_force(force, disp)
        F_01_i = np.searchsorted(force[:F_max_i], F_max*0.1, side='left') #Note these only work because the subset is at the start of the array
        F_04_i = np.searchsorted(force[:F_max_i], F_max*0.4, side='left')
        
        if (force[F_01_i+1] - force[F_01_i]) == 0:
            d_10 = disp[F_01_i]
        else:
            d_10 = (disp[F_01_i+1] - disp[F_01_i]) / (force[F_01_i+1] - force[F_01_i]) * (F_max*0.1-force[F_01_i]) + disp[F_01_i]
        if (force[F_04_i+1] - force[F_04_i]) == 0:
            d_40 = disp[F_04_i]
        else:
            d_40 = (disp[F_04_i+1] - disp[F_04_i]) / (force[F_04_i+1] - force[F_04_i]) * (F_max*0.4-force[F_04_i]) + disp[F_04_i]
        
        #print(f"d10, {d_10}")
        #print(f"d40, {d_40}")
        
        
        alpha_grad = (F_max*0.4 - F_max*0.1) / (d_40 - d_10)
        alpha_intercept = force[F_01_i] - alpha_grad * disp[F_01_i]
        self.values['stiffness_10_40'] = alpha_grad
        return alpha_grad, alpha_intercept
    #end
    
    
    def elastic_stiffness(self, min_F = 0.1, max_F = 0.2):
        if not 'interp_backbone' in self.curves:
            self.backbone()
            self.interpolate_backbone()
        force = self.curves['interp_backbone']['force']
        disp = self.curves['interp_backbone']['disp']
        F_max, V_F_max, F_max_i = max_force(force, disp)
        alpha_grads = []
        alpha_intercepts = []
        r_2s = []
        i_list = np.arange(min_F, max_F+0.05, 0.05)
        for i in i_list:
            F_low_i = np.searchsorted(force[:F_max_i], F_max*i, side='left') #Note these only work because the subset is at the start of the array
            F_high_i = np.searchsorted(force[:F_max_i], F_max*(i+0.3), side='left')
            alpha_grad = (force[F_high_i] - force[F_low_i]) / (disp[F_high_i] - disp[F_low_i])
            alpha_intercept = force[F_low_i] - alpha_grad * disp[F_low_i]
            
            x = disp[F_low_i: F_high_i+1]
            y = force[F_low_i: F_high_i+1]
           
            pred = alpha_intercept + alpha_grad * x
            res = y - pred
            ss_res = np.sum(res**2)
            ss_tt = np.sum((y-np.mean(x))**2)
            r_2 = 1 - ss_res/ss_tt
            
            r_2s.append(r_2)
            alpha_grads.append(alpha_grad)
            alpha_intercepts.append(alpha_intercept)            
        #minimise r2
        max_r2_i = np.argmax(r_2s)
        self.values['stiffness'] = alpha_grads[max_r2_i]
        self.values['intercept'] = alpha_intercepts[max_r2_i]
        self.values['stiffness_r2'] = r_2s[max_r2_i]
        self.values['stiffness_start'] = i_list[max_r2_i]
        return alpha_grads[max_r2_i], alpha_intercepts[max_r2_i]
    
    
    def max_force(self):
        """ 
        Calculates the max force and displacement at max force
        Written by TW 29/06/20
        Inputs:
            -force
            -displacement
        Outputs:
            -F_max
            -V_F_max
            -F_max_i
        """
        if not 'backbone' in self.curves:
            self.backbone()
        
        force=self.curves['backbone']['force'].to_numpy()
        disp=self.curves['backbone']['disp'].to_numpy()
        F_max, V_F_max, F_max_i = max_force(force, disp)
        # F_max = np.amax(force)
        # F_max_i  = np.argmax(force)
        # V_F_max = disp[F_max_i]
        self.values['F_max'] = F_max
        self.values['V_F_max'] = V_F_max
        return F_max, V_F_max, F_max_i
    #end 
        
        
    def EN12512_ductility(self, display_values=False, plot=True):
        """ 
        Calculates the ductility based on EN12512 method
        Written by TW 29/06/20
        Inputs:
            -force
            -displacement
            -axes
            -display_values
        Outputs:
            -F_y
            -V_y
            -mu 
        """
        if not 'interp_backbone' in self.curves:
            self.backbone()
            self.interpolate_backbone()
        try:
            alpha_grad = self.values['stiffness']
            alpha_intercept = self.values['intercept']
        except Error:
            alpha_grad, alpha_intercept = self.elastic_stiffness()
            
        force=self.curves['interp_backbone']['force'].to_numpy()
        disp=self.curves['interp_backbone']['disp'].to_numpy()
        F_max, V_F_max, F_max_i = max_force(force, disp)
        
        beta_grad = 1/6 * alpha_grad
        cur_beta_intercept = 0
        cur_beta_intercept_i = 0
        for i in range(0, F_max_i):
            beta_intercept = force[i] - beta_grad * disp[i]
            if beta_intercept > cur_beta_intercept:
                cur_beta_intercept = beta_intercept
                cur_beta_intercept_i = i
            #end
        #end
        beta_intercept = cur_beta_intercept                
        V_y = (beta_intercept - alpha_intercept) / (alpha_grad - beta_grad)
        F_y = alpha_grad * V_y + alpha_intercept
        F_y_i = np.searchsorted(force[:F_max_i], F_y, side='left')
        F_u, V_u, F_u_i  = self.ultimate_displacement()
        mu = V_u / V_y
        
        if plot:
            fig_EN12512 = plt.subplots()
            fig_EN12512[1].plot(self.data.disp, self.data.force, 'c') #Plot force
            fig_EN12512[1].plot(disp, force, 'r', label='backbone') #Plot backbone
            fig_EN12512[1].plot(disp[:F_y_i], alpha_grad * disp[:F_y_i] + alpha_intercept , label=f'alpha = {alpha_grad:0.2f}x+{alpha_intercept:0.2f}')
            fig_EN12512[1].plot(disp[:cur_beta_intercept_i], beta_grad * disp[:cur_beta_intercept_i] + beta_intercept, label=f'beta = {beta_grad:0.2f}x+{beta_intercept:0.2f}')
            fig_EN12512[1].plot(disp[F_max_i:], np.ones(len(disp[F_max_i:])) * F_u, label=f'F_u= {F_u:0.2f} kN')
            fig_EN12512[1].set_title(self.name+'\nEN12512')
            fig_EN12512[1].set_xlabel('Displacement (mm)')
            fig_EN12512[1].set_ylabel('Force (kN)')
            fig_EN12512[1].legend(loc=4)
            self.plots['EN12512'] = fig_EN12512
            
        if display_values:
            print('{:20}={:10.2f} {}'.format('F_max', F_max, 'kN'))
            print('{:20}={:10.2f} {}'.format('alpha_grad', alpha_grad, 'kN/mm'))
            print('{:20}={:10.2f} {}'.format('alpha_intercept', alpha_intercept, 'kN'))
            print('{:20}={:10.2f} {}'.format('beta_grad', beta_grad, 'kN/mm'))
            print('{:20}={:10.2f} {}'.format('beta_intercept', beta_intercept, 'kN'))
            print('{:20}={:10.2f} {}'.format('F_y', F_y, 'kN'))
            print('{:20}={:10.2f} {}'.format('V_y', V_y, 'mm'))
            print('{:20}={:10.2f} {}'.format('F_u', F_u, 'kN'))
            print('{:20}={:10.2f} {}'.format('V_u', V_u, 'mm'))
            print('{:20}={:10.2f} {}'.format('mu', mu, ''))
            
        self.values['F_y_EN12512'] = F_y
        self.values['V_y_EN12512'] = V_y
        self.values['mu_EN12512'] = mu
        return F_y, V_y, mu
    #end      
       
        
    def EEEP_ductility(self, display_values=False, plot=True):
        """ 
        Calculates the ductility based on EEEP method
        Written by TW 29/06/20
        Inputs:
            -force
            -displacement
            -axes
            -display_values
        Outputs:
            -F_y
            -V_y
            -mu
        """
        if not 'backbone' in self.curves:
            self.backbone()
        force=self.curves['backbone']['force'].to_numpy()
        disp=self.curves['backbone']['disp'].to_numpy()
        try:
            alpha_grad = self.values['stiffness']
            alpha_intercept = self.values['intercept']
        except Error:
            alpha_grad, alpha_intercept = self.elastic_stiffness()
        F_max, V_F_max, F_max_i = max_force(force, disp)

        F_u, V_u, F_u_i  = self.ultimate_displacement()
        A = 0
        for i in range(1, len(force[:F_u_i])):
            A += (force[i] + force[i-1])/2 * abs(disp[i] - disp[i-1])
        #end
        V_y = V_u - np.sqrt((V_u ** 2) - 2 * A / alpha_grad)
        F_y = V_y * alpha_grad
        V_y_i = np.searchsorted(disp[:F_max_i], V_y, side='left')
        mu = V_u / V_y
        
        if plot:
            fig_EEEP = plt.subplots()
            fig_EEEP[1].plot(self.data.disp, self.data.force, 'c') #Plot force
            fig_EEEP[1].plot(disp, force, 'r', label='Backbone') #Plot backbone
            fig_EEEP[1].plot(disp[:V_y_i+1], alpha_grad * disp[:V_y_i+1] , label=f'alpha = {alpha_grad:0.2f}x+{alpha_intercept:0.2f}') #+ alpha_intercept
            fig_EEEP[1].plot(disp[:V_y_i], np.ones(len(disp[:V_y_i])) * F_y, label=f'F_y = {F_y:0.2f} kN')
            fig_EEEP[1].plot(disp[F_max_i:], np.ones(len(disp[F_max_i:])) * F_u, label=f'F_u= {F_u:0.2f} kN')
            fig_EEEP[1].set_title(self.name+'\nEEEP')
            fig_EEEP[1].set_xlabel('Displacement (mm)')
            fig_EEEP[1].set_ylabel('Force (kN)')
            fig_EEEP[1].legend(loc=4)
            self.plots['EEEP'] = fig_EEEP
        #end
        if display_values:
            print('{:20}={:10.2f} {}'.format('F_max', F_max, 'kN'))
            print('{:20}={:10.2f} {}'.format('alpha_grad', alpha_grad, 'kN/mm'))
            print('{:20}={:10.2f} {}'.format('alpha_intercept', alpha_intercept, 'kN'))
            print('{:20}={:10.2f} {}'.format('A', A, 'kNmm'))
            print('{:20}={:10.2f} {}'.format('F_y', F_y, 'kN'))
            print('{:20}={:10.2f} {}'.format('V_y', V_y, 'mm'))
            print('{:20}={:10.2f} {}'.format('F_u', F_u, 'kN'))
            print('{:20}={:10.2f} {}'.format('V_u', V_u, 'mm'))
            print('{:20}={:10.2f} {}'.format('mu', mu, ''))
            
            
        self.values['F_y_EEEP'] = F_y
        self.values['V_y_EEEP'] = V_y
        self.values['mu_EEEP'] = mu
        #self.values['A_EEEP'] = A
        return F_y, V_y, mu
    #end
    
    
    def offset_ductility(self, d=12, display_values=False, plot=True, offset = 0.05):
        """ Does something """
        if not 'interp_backbone' in self.curves:
            self.backbone()
            self.interpolate_backbone()
        force=self.curves['interp_backbone']['force'].to_numpy()
        disp=self.curves['interp_backbone']['disp'].to_numpy()
        F_max, V_F_max, F_max_i = max_force(force, disp)
        try:
            alpha_grad = self.values['stiffness']
            alpha_intercept = self.values['intercept']
        except Error:
            alpha_grad, alpha_intercept = self.elastic_stiffness()
        
        alpha_intercept_offset = alpha_intercept - alpha_grad * ( offset * d)
        
        cur_offset_intercept = 0
        cur_offset_intercept_i = 0
        i = F_max_i
        flag = True
        while flag:        
            cur_disp = disp[i]
            force_curve = force[i]
            force_line = alpha_grad * cur_disp + alpha_intercept_offset
            if force_curve > force_line:
                flag = False
            i -= 1 # increment it after
        V_y = disp[i]
        F_y = alpha_grad * V_y + alpha_intercept_offset
        F_u, V_u, F_u_i  = self.ultimate_displacement()
        mu = V_u/V_y
        
        V_y_i = np.searchsorted(disp[:F_max_i], V_y, side='left')
        if plot:
            fig_offset = plt.subplots()
            fig_offset[1].plot(self.data.disp, self.data.force, 'c') #Plot force
            fig_offset[1].plot(disp, force, 'r', label='backbone') #Plot backbone
            fig_offset[1].plot(disp[:V_y_i+1], alpha_grad * disp[:V_y_i+1] + alpha_intercept, label=f'alpha = {alpha_grad:0.2f}x+{alpha_intercept:0.2f}') #+ alpha_intercept
            fig_offset[1].plot(disp[:V_y_i+1], alpha_grad * disp[:V_y_i+1] +alpha_intercept_offset, label=f'offset = {alpha_grad:0.2f}x+{alpha_intercept_offset:0.2f}')
            fig_offset[1].plot(disp[:V_y_i], np.ones(len(disp[:V_y_i])) * F_y, label=f'F_y = {F_y:0.2f} kN')
            fig_offset[1].plot(disp[F_max_i:], np.ones(len(disp[F_max_i:])) * F_u, label=f'F_u= {F_u:0.2f} kN')
            fig_offset[1].set_title(self.name+f'\nOffset {offset} x d')
            fig_offset[1].set_xlabel('Displacement (mm)')
            fig_offset[1].set_ylabel('Force (kN)')
            fig_offset[1].legend(loc=4)
            self.plots['offset'] = fig_offset
        #end
        
        self.values['F_y_offset'] = F_y
        self.values['V_y_offset'] = V_y
        self.values['mu_offset'] = mu      
        return F_y, V_y, mu
    #end
    
    
    def force_at_displacement(self, plot=True, disp_value=5):
        """ Does something """
        if not 'interp_backbone' in self.curves:
            self.backbone()
            self.interpolate_backbone()
        force=self.curves['interp_backbone']['force'].to_numpy()
        disp=self.curves['interp_backbone']['disp'].to_numpy()
        F_max, V_F_max, F_max_i = max_force(force, disp)
        
        i = np.searchsorted(disp, disp_value, side='left') #Using the interp_backbone, so dont need to use interpolation
        F = force[i]
        
        if plot:
            fig_disp = plt.subplots()
            fig_disp[1].plot(self.data.disp, self.data.force, 'c') #Plot force
            fig_disp[1].plot(disp, force, 'r', label='backbone') #Plot backbone
            fig_disp[1].plot(np.arange(0, disp_value), np.ones(len(np.arange(0, disp_value))) * F, label=f'{F:0.2f} kN')
            fig_disp[1].plot(np.ones(len(np.arange(0, F_max))) * disp_value, np.arange(0, F_max), label=f'{disp_value:0.2f} mm')
            fig_disp[1].set_title(self.name+f'\nForce at {disp_value} mm')
            fig_disp[1].set_xlabel('Displacement (mm)')
            fig_disp[1].set_ylabel('Force (kN)')
            fig_disp[1].legend(loc=4)
            self.plots['force_at_{disp_value}mm'] = fig_disp

        self.values[f'force_at_{disp_value}mm'] = F
        return F, disp_value
        
    
    def force_up_to_displacement(self, plot=True, disp_value=5):
        """ Does something """
        if not 'interp_backbone' in self.curves:
            self.backbone()
            self.interpolate_backbone()
        force=self.curves['interp_backbone']['force'].to_numpy()
        disp=self.curves['interp_backbone']['disp'].to_numpy()
        
        
        i = np.searchsorted(disp, disp_value, side='left') #Using the interp_backbone, so dont need to use interpolation
        F_max, V_F_max, F_max_i = max_force(force[:i], disp[:i])
        
        
        if plot:
            fig_disp = plt.subplots()
            fig_disp[1].plot(self.data.disp, self.data.force, 'c') #Plot force
            fig_disp[1].plot(disp, force, 'r', label='backbone') #Plot backbone
            fig_disp[1].plot(np.arange(0, disp_value), np.ones(len(np.arange(0, disp_value))) * F_max, label=f'{F_max:0.2f} kN')
            fig_disp[1].plot(np.ones(len(np.arange(0, F_max))) * V_F_max, np.arange(0, F_max), label=f'{V_F_max:0.2f} mm')
            fig_disp[1].set_title(self.name+f'\nMax Force up to {disp_value} mm')
            fig_disp[1].set_xlabel('Displacement (mm)')
            fig_disp[1].set_ylabel('Force (kN)')
            fig_disp[1].legend(loc=4)
            self.plots['force_up_to_{disp_value}mm'] = fig_disp

        self.values[f'force_up_to_{disp_value}mm'] = F_max
        return F_max, V_F_max
        
    
    def v_eq(self):
        cycle_peak_disps = []
        E_ps = []
        E_ds = []
        v_eqs = []
        E_d_graph = []
        cumulative_disp_graph= []
        cumulative_disp = 0
        E_d_total = 0
        for cycle in self.cycles:
            peak = cycle.find_peak() #Re find peak now that data has been smoothed. May not be necessary. Probably not necessary
            peak = cycle.peak
            E_p = peak['force'] * peak['disp'] / 2  
            E_d = 0
            for i in range(0, len(cycle.forward) -1):
                cumulative_disp += abs(cycle.forward['disp'].iloc[i+1] - cycle.forward['disp'].iloc[i])
                E_d_total += (cycle.forward['force'].iloc[i+1] + cycle.forward['force'].iloc[i]) / 2 * (cycle.forward['disp'].iloc[i+1] - cycle.forward['disp'].iloc[i])
                E_d += (cycle.forward['force'].iloc[i+1] + cycle.forward['force'].iloc[i]) / 2 * (cycle.forward['disp'].iloc[i+1] - cycle.forward['disp'].iloc[i])
                cumulative_disp_graph.append(cumulative_disp)
                E_d_graph.append(E_d_total)
                
            for i in range(0, len(cycle.back) -1):
                cumulative_disp += abs(cycle.back['disp'].iloc[i+1] - cycle.back['disp'].iloc[i])
                E_d_total += (cycle.back['force'].iloc[i+1] + cycle.back['force'].iloc[i]) / 2 * (cycle.back['disp'].iloc[i+1] - cycle.back['disp'].iloc[i])
                E_d += (cycle.back['force'].iloc[i+1] + cycle.back['force'].iloc[i]) / 2 * (cycle.back['disp'].iloc[i+1] - cycle.back['disp'].iloc[i])
                cumulative_disp_graph.append(cumulative_disp)
                E_d_graph.append(E_d_total)
            v_eq1 = E_d/(2*3.141*E_p) *100 #I think Ep and Ed are round the wrong way in the standard diagram
            
            E_ps.append(E_p)
            E_ds.append(E_d)
            v_eqs.append(v_eq1)
            cycle_peak_disps.append(peak['disp'])
            
        d = {'Peak_disp': cycle_peak_disps ,'E_d': E_ds, 'E_p': E_ps, 'v_eq': v_eqs} 
        self.curves['energy_dissipation'] = pd.DataFrame(d)
            
        d = {'E_d': E_d_graph, 'Cumulative Disp': cumulative_disp_graph} 
        self.curves['cumulative_Ed'] = pd.DataFrame(d)          
        return E_d_graph, cumulative_disp_graph
    
    
    def ultimate_displacement(self):
        """ 
        Calculates the ultiamte displacement based on 0.8 F_max
        Written by TW 29/06/20
        Inputs:
            -force
            -displacement
        Outputs:
            -F_u
            -V_u
            -F_u_i
        """
        if not 'backbone' in self.curves:
            self.backbone()
        force=self.curves['backbone']['force'].to_numpy()
        disp=self.curves['backbone']['disp'].to_numpy()
        F_max, V_F_max, F_max_i = max_force(force, disp)
        F_u = F_max * 0.8
        F_u_i = np.searchsorted(force[:F_max_i:-1], F_u, side='right') #Used right as works backwards
        
        
        
        
        
        
        #V_u = disp[:F_max_i:-1][F_u_i] Changed from this to a 
        
        F_u_i = len(force) - F_u_i -1 #THis needs to be reworked. Have only fixed to allow eeep method to work
        try: #Added a try except incase failiure mode is brtittle and 0.8 F max is never achieved
            if (force[F_u_i+1] - force[F_u_i]) == 0:
                V_u = disp[F_u_i]
            else:
                V_u = (disp[F_u_i+1] - disp[F_u_i]) / (force[F_u_i+1] - force[F_u_i]) * (F_u-force[F_u_i]) + disp[F_u_i]
        except:
            V_u = disp[F_u_i]
        
        self.values['F_u'] = F_u
        self.values['V_u'] = V_u
        #self.values['F_u_i'] = F_u_i
        return F_u, V_u, F_u_i 
    #end
    
    
    def interpolate_backbone(self, x_step = 0.01, backbone_curve='backbone'): #Roughly 4000 points for backbone
        """ Note x and y need to be backbone values"""
        x = self.curves[backbone_curve]['disp']
        y = self.curves[backbone_curve]['force']
        max_x = round(max(x) / x_step) * x_step
        min_x = min(round(min(x) / x_step) * x_step, 0) # changed from 0 on 6/11/22
        new_x = np.arange(min_x, max_x, x_step)
        f = interp1d(x, y, 'linear', fill_value='extrapolate')
        new_y = f(new_x)
        
        d = {'force': new_y, 'disp': new_x} 
        # np.array([force_backbone, disp_backbone]).T, columns=['force', 'disp']
        self.curves['interp_backbone'] = pd.DataFrame(d)
        
        
    def trim_start(self, disp_threshold, force_threshold):
        pass #May not be needed as cycle processing does this already
        # if len(self.data['control'] > 1):
        #     i = 1
        #     while self.data['control']
            
        #     self.data = 
            
    def envelope_curves(self):
        cycle_peak_disps = []
        max_forces = []
        max_force_disps = []
        for cycle in self.cycles:

            
            peak = cycle.find_peak() #Re find peak now that data has been smoothed. May not be necessary. Probably not necessary
            peak = cycle.peak
            max_force, max_force_disp = cycle.max_force()
            
            
            max_forces.append(max_force)
            max_force_disps.append(max_force_disp)
            cycle_peak_disps.append(peak['disp'])
            
        d = {'Peak_disp': cycle_peak_disps, 'max_force': max_forces, 'max_force_disp': max_force_disps} 
        self.curves['envelope_curves'] = pd.DataFrame(d)       
        
        
        
        
def max_force(force, disp): #Not sure if this is still needed. Made to use a passed force instead of normal force as indexs dont line up between curve and 
    """Expects force and disp as numpy arrays or similar"""
    F_max = np.amax(force)
    F_max_i  = np.argmax(force)
    V_F_max = disp[F_max_i]
    return F_max, V_F_max, F_max_i