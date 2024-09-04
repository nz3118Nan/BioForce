#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2024/09/04 13:57:56
@Author  :   Nan Zhou
@Version :   0.3
@Desc    :   None
'''
#############################################



import os 

v = 'v4'
os.system(f"python Academic/A01_RIF/method/data_process.py --data_version RIF_V2 --exp_version {v} --num_val 25 --target label_1 --description 'n400 val 50 随机挑选 label_1'")
os.system(f"python Academic/A01_RIF/method/result_generation.py --exp_version {v}")
os.system(f"python Academic/A01_RIF/method/evaluation_plot.py --exp_version {v}")

