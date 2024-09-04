#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   __initi__.py
@Time    :   2024/09/03 17:24:53
@Author  :   Nan Zhou
@Version :   0.3
@Desc    :   None
'''
####################################################################
# system setting
import os
import sys
file_path = os.path.abspath(__file__)
root_dir = os.path.abspath(os.path.join(file_path, '..', '..'))
sys.path.append(root_dir)
sys.dont_write_bytecode = True
os.chdir(root_dir)

# build-in package 
import pandas as pd
import numpy as np
####################################################################

file_list = ['Academic', 'Commercial']

for file_path in file_list:
    file_path = os.path.join(root_dir, file_path)
    
    for file_folder in os.listdir(file_path):
        # check file)folder is in BigData folder or not 
        ## if not, create a new folder in BigData named as file_folder
        ## if yes, pass
        
        if file_folder not in os.listdir('BigData'):
            os.mkdir(os.path.join('BigData', file_folder))
        else:
            pass