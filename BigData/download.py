#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   download.py
@Time    :   2024/09/03 20:44:50
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
import gdown
import json

####################################################################

    

# BigData/bigdata_info.json
## load the bigdata_info.json
with open('BigData/bigdata_info.json') as f:
    bigdata_info_json = json.load(f)
    

def download_data(file_name):
    file_id = bigdata_info_json[file_name]['link'].split('/')[-2]
    output = bigdata_info_json[file_name]['store_path']

    if not os.path.exists(output):
        gdown.download(f'https://drive.google.com/uc?id={file_id}', output, quiet=False)

        # check the file exists
        print(f"Check the file exists: {os.path.exists(output)}")

    else:
        print("The file exists")
        
    return output

####################################################################
if __name__ == '__main__':
    file_name = 'RIF_V1'
    output = download_data(file_name)
    print(output)