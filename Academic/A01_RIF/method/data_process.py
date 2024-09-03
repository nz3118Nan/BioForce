#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   data_process.py
@Time    :   2024/09/03 21:38:07
@Author  :   Nan Zhou
@Version :   0.3
@Desc    :   None
'''
#######################################

# system setting
import os
import sys
file_path = os.path.abspath(__file__)
root_dir = os.path.abspath(os.path.join(file_path, '..', '..', '..'))
sys.path.append(root_dir)
sys.dont_write_bytecode = True
os.chdir(root_dir)

# build-in package 
import pandas as pd
import numpy as np
import json

# developed package
from utils.func import MS  

#######################################