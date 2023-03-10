# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 23:15:29 2021

@author: TomsS
"""

import numpy as np

def max_force_backbone(force, disp):
    """ Is it bigger than we've ever seen before """
    force_backbone = []
    disp_backbone = []
    if not len(force) == 0:
        cur_force = force[0]
    #cur_disp = np.amax(disp, axis=0)
    for i in range(0, len(force)):
        if force[i] >= cur_force:
            force_backbone.append(force[i])
            disp_backbone.append(disp[i])
            cur_force = force[i]
        #end
    #end
    return force_backbone, disp_backbone
#end
            

def backbone(force, disp):
    """ """
   
    increasing_disp_indicies = np.argwhere(np.diff(disp) > 0).flatten() + 1 #Had to add flatten as argwhere produces a column vector
    increasing_disp_indicies = np.insert(increasing_disp_indicies, 0, 0)
    
   #print('hh',increasing_disp_indicies)
    # force = force[[0]+increasing_disp_indicies] 
    # force = force[[increasing_disp_indicies[0]-1]+increasing_disp_indicies] #Only use parts of curve where displacement is increasing
    # disp = disp[[increasing_disp_indicies[0]-1]+increasing_disp_indicies] #A slightly more robust version might go through displacements and make sure they are bigger than the last
    force = force[increasing_disp_indicies]
    disp = disp[increasing_disp_indicies]
    
    F_max_index = np.argmax(force)
    disp_max_index = np.argmax(disp)
     
    force_backbone_pre_max, disp_backbone_pre_max = max_force_backbone(force[:F_max_index], disp[:F_max_index])
    force_backbone_post_max, disp_backbone_post_max = max_force_backbone(force[disp_max_index:F_max_index-1:-1], disp[disp_max_index:F_max_index-1:-1])
    
    # Should have a section to remove any overlaps
    force_backbone = np.concatenate((force_backbone_pre_max ,force_backbone_post_max[::-1]))
    disp_backbone = np.concatenate((disp_backbone_pre_max, disp_backbone_post_max[::-1]))
    
    
    increasing_disp_indicies = np.argwhere(np.diff(disp_backbone) > 0).flatten() + 1 #Had to add flatten as argwhere produces a column vector
    increasing_disp_indicies = np.insert(increasing_disp_indicies, 0, 0)
    disp_backbone = disp_backbone[increasing_disp_indicies]
    force_backbone = force_backbone[increasing_disp_indicies]
    
    return force_backbone, disp_backbone
#end


def max_force_at_displacement_backbone(force, disp):
    """ Is it bigger than we've ever seen before """
    force_backbone = []
    disp_backbone = []
    if not len(force) == 0:
        cur_disp = disp[0]
    #cur_disp = np.amax(disp, axis=0)
    for i in range(0, len(disp)):
        if disp[i] >= cur_disp:
            force_backbone.append(force[i])
            disp_backbone.append(disp[i])
            cur_disp = disp[i]
        #end
    #end
    return force_backbone, disp_backbone
#end


def backbone2(force, disp):
    """ """
   
    increasing_disp_indicies = np.argwhere(np.diff(disp) > 0).flatten() + 1 #Had to add flatten as argwhere produces a column vector
    increasing_disp_indicies = np.insert(increasing_disp_indicies, 0, 0)
    
    force = force[increasing_disp_indicies]
    disp = disp[increasing_disp_indicies]
    
    F_max_index = np.argmax(force)
    disp_max_index = np.argmax(disp)
     
    force_backbone, disp_backbone = max_force_at_displacement_backbone(force, disp)
    # force_backbone_post_max, disp_backbone_post_max = max_force_backbone(force[disp_max_index:F_max_index-1:-1], disp[disp_max_index:F_max_index-1:-1])
    
    # # Should have a section to remove any overlaps
    # force_backbone = np.concatenate((force_backbone_pre_max ,force_backbone_post_max[::-1]))
    # disp_backbone = np.concatenate((disp_backbone_pre_max, disp_backbone_post_max[::-1]))
    
    
    # increasing_disp_indicies = np.argwhere(np.diff(disp_backbone) > 0).flatten() + 1 #Had to add flatten as argwhere produces a column vector
    # increasing_disp_indicies = np.insert(increasing_disp_indicies, 0, 0)
    # disp_backbone = disp_backbone[increasing_disp_indicies]
    # force_backbone = force_backbone[increasing_disp_indicies]
    
    return force_backbone, disp_backbone
#end


def backbone3(cycles, order = 5):
    # force_backbone = []
    # disp_backbone = []
    
    first_cycle = cycles[0]
    force = first_cycle.forward['force'].tolist()
    disp = first_cycle.forward['disp'].tolist()
    force_backbone, disp_backbone = max_force_at_displacement_backbone(force[:-1], disp[:-1]) #note I changed this after running
    
    for cycle in cycles[1:]:
        force = cycle.forward['force'].tolist()
        disp = cycle.forward['disp'].tolist()
        i = 0
        while i < len(disp) and disp_backbone[-1] > disp[i]:# skip to where the last cycle left off
            i += 1
        while i < len(force) and force_backbone[-1] > force[i] and not np.all(force[i] > np.array(force[i+1:i+1+order])):
            i += 1
        

        if i + 5 < len(force):
            # print("Cycle Number", cycle.cycle_number)
            # print(disp[i], force_backbone[-1], force[i])
            # print(i < len(force), force_backbone[-1] > force[i], not np.all(force[i] > np.array(force[i+1:i+1+order])))
            f_backbone, d_backbone = max_force_at_displacement_backbone(force[i:-1], disp[i:-1]) #Note this has been added to ensure that if the last data point was a drop, it will not be accounted for
            force_backbone += f_backbone
            disp_backbone += d_backbone
            # print(f_backbone)
            # print()

    # force_backbone = df.Series(force_backbone)
    # disp_backbone = df.Series(disp_backbone)
    return force_backbone, disp_backbone