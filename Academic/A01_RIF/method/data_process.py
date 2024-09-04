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
root_dir = os.path.abspath(os.path.join(file_path, '..', '..', '..', '..'))
sys.path.append(root_dir)
sys.dont_write_bytecode = True
os.chdir(root_dir)

# build-in package 
import pandas as pd
import numpy as np
import json
from argparse import ArgumentParser
from tqdm import tqdm

# developed package
from Academic.A01_RIF.utils.func import MS  
from BigData.download import download_data
#######################################

# set a argument parser name data_version, need to be string
parser = ArgumentParser()
parser.add_argument("--data_version", type=str, default='RIF_V1', help="data version")
parser.add_argument("--exp_version", type=str, default='v1', help="exp version")    
parser.add_argument("--num_val", type=int, default=10, help="number of validation") 
parser.add_argument("--target", type=str, default='label_1', help="target column") 
parser.add_argument("--description", type=str, default='', help="description") 

# get data_version
args = parser.parse_args()

# get data_version
data_version = args.data_version
v = args.exp_version
n = args.num_val
target = args.target
description = args.description

# convert to upper case
version = v.upper()


# BigData/bigdata_info.json
with open(os.path.join(root_dir, 'BigData/bigdata_info.json'), 'r') as f:
    bigdata_info = json.load(f)
    
data_path = os.path.join(root_dir, bigdata_info[data_version]['store_path'])

# check if data exist
if not os.path.exists(data_path):
    download_data(data_version)
else:
    print(f"{data_version} exists")

data = pd.read_csv(data_path, index_col=0)


# rename '日期差值' to interval_from_ert_to_transfer
# rename 基础内分泌AMH to AMH
# drop  '基础内分泌FSH', '基础内分泌E2','基础内分泌LH'
# data.rename(columns={'日期差值':'interval_from_ert_to_transfer'}, inplace=True)
data.drop(['基础内分泌FSH', '基础内分泌E2','基础内分泌LH','基础内分泌AMH'], axis=1, inplace=True)

data.drop(['day_of_embryo'], axis=1, inplace=True)

# rename '移植胚胎评价' to embryo_evaluation
# data.rename(columns={'移植胚胎评价':'embryo_evaluation'}, inplace=True)

# drop  '基础内分泌FSH', '基础内分泌E2', '基础内分泌LH', '基础内分泌AMH',
# data.drop(['基础内分泌FSH', '基础内分泌E2', '基础内分泌LH', '基础内分泌AMH'], axis=1, inplace=True)


embryo_evaluation = list(data['移植胚胎评价'])

for i in range(len(embryo_evaluation)):
    tmp = embryo_evaluation[i]
    
    # if tmp is X月Y日, change to Y/X
    if '月' in tmp and '日' in tmp:
        month = tmp.split('月')[0]
        day = tmp.split('月')[1].split('日')[0]
        embryo_evaluation[i] = f'{day}/{month}'
        
    if ',' in tmp:
        embryo_evaluation[i] = tmp.split(',')
    
    if '，' in tmp:
        embryo_evaluation[i] = tmp.split('，')
        
    if ' ' in tmp:
        # '  ' to ' '
        # '   ' to ' '
        # '    ' to ' '
        tmp = tmp.replace('  ', ' ')
        tmp = tmp.replace('   ', ' ')
        tmp = tmp.replace('    ', ' ')
        
        embryo_evaluation[i] = tmp.split(' ')
        
for i in range(len(embryo_evaluation)):
    if type(embryo_evaluation[i]) != list:
        embryo_evaluation[i] = [embryo_evaluation[i]]
        
    else:
        tmp = embryo_evaluation[i]
        
        for j in range(len(tmp)):
            if tmp[j] == '':
                tmp.pop(j)
                break
        embryo_evaluation[i] = tmp
        
data['embryo_evaluation_modified'] = embryo_evaluation
        
# 4AA、4BB、4AB、4BA if appears, add 0.2 to the score. Appears 2 times add 0.2 still. 
score_list = [0] * len(data)
for i in range(len(data)):
    embryo_evaluation = data['embryo_evaluation_modified'].iloc[i]
    
    # any element in embryo_evaluation is 4AA, 4BB, 4AB, 4BA, add 0.2 to the score
    add = 0
    for item in embryo_evaluation:
        if item == '4AA' or item == '4BB' or item == '4AB' or item == '4BA' \
            or '5AA' or item == '5BB' or item == '5AB' or item == '5BA' \
            or item == '6AA' or item == '6BB' or item == '6AB' or item == '6BA' \
            or item == '11/7' or item == '12/7' or item == '21/7'\
            or item == '11/8' or item == '12/8' or item == '21/8'\
            or item == '11/9' or item == '12/9' or item == '21/9':
            add += 0.2
        else:
            add += 0.1 
    
    score_list[i] += add 
    
    # add number of embryo_evaluation_num
    
    score_list[i] += data['num_of_embryo'].iloc[i]
    
data['embryo_evaluation_value'] = score_list

# drop embryo_evaluation and embryo_evaluation_modified
data.drop(['移植胚胎评价', 'embryo_evaluation_modified'], axis=1, inplace=True)
        

data['interval_from_ert_to_transfer'] = data['日期差值']

data.drop(['日期差值'], axis=1, inplace=True)

移植日期 = data['移植日期']

# convert to datetime
移植日期 = pd.to_datetime(移植日期, format='%Y-%m-%d')

检测日期 = data['检测流水号']
检测日期 = 检测日期.str.split('_').str[2]
检测日期 = ['20' + i for i in 检测日期]

# convert to datetime
检测日期 = pd.to_datetime(检测日期, format='%Y%m%d')

data['检测日期'] = 检测日期
data['移植日期'] = 移植日期


data.dropna(inplace=True)
print(f"Number of rows: {data.shape[0]}")

data_0 = data[data[target] == 0]
data_1 = data[data[target] == 1]

# random select n data_0 and data_1 into data_val 
import random
random_seed = random.randint(0, 1000)
data_0_val = data_0.sample(n=n, random_state=random_seed)
data_1_val = data_1.sample(n=n, random_state=random_seed)



data_0 = data[data[target] == 0]
data_1 = data[data[target] == 1]

# random select n data_0 and data_1 into data_val 
import random
random_seed = random.randint(0, 1000)
data_0_val = data_0.sample(n=n, random_state=random_seed)
data_1_val = data_1.sample(n=n, random_state=random_seed)

# # sort by 移植日期 for data_0 and data_1
# data_0 = data_0.sort_values(by='移植日期', ascending=False)
# data_1 = data_1.sort_values(by='移植日期', ascending=False)

# data_0.reset_index(drop=True, inplace=True)
# data_1.reset_index(drop=True, inplace=True)

# get the first n samples for data_0 and data_1
# data_0_val = data_0.iloc[:n]
# data_1_val = data_1.iloc[:n]

# data_0 = data_0.iloc[n:]
# data_1 = data_1.iloc[n:]

# data_0 drop data_0_val
data_0.drop(data_0_val.index, axis=0, inplace=True)
data_1.drop(data_1_val.index, axis=0, inplace=True)

data_val = pd.concat([data_0_val, data_1_val], axis=0)
data_val.reset_index(drop=True, inplace=True)

data = pd.concat([data_0, data_1], axis=0)
data.reset_index(drop=True, inplace=True)

# sort by 移植日期 for data_0 and data_1
data = data.sort_values(by='移植日期', ascending=False)
data_val = data_val.sort_values(by='移植日期', ascending=False)

data.reset_index(drop=True, inplace=True)
data_val.reset_index(drop=True, inplace=True)

# count the number of target in each group
print(data_val[target].value_counts())
print(data[target].value_counts())

# print shape
print(data_val.shape)
print(data.shape)


# target feature
target_list = ['label_1', 'label_2', 'label_3', ]
col_list_drop = ['移植日期',
 '检测流水号',
 '检测结果',
 'KEYCOD',
 'REGCOD',
 '检测日期']
col_list_drop += target_list
col_list_drop.remove(target)
data = data.drop(col_list_drop, axis=1)


# target feature
# 检测结果 == 容受期
# data = data[data['检测结果'] == '容受期']

# drop 检测结果
# data = data.drop(['检测结果'], axis=1)

# clinical feature
clinical_feature = ['BMI', 'AMH','num_of_embryo', 'embryo_evaluation_value'] # 'Age_at_embryo_transfer', 
data_clinical = data[clinical_feature]

# basic info
X = data.drop([target], axis=1)
## data shape
print(f"Data shape: {X.shape}")

# for each col, calculate the least 95% quantile mean value, get rid of the largest 5% value
mean_list = []
for col in tqdm(X.columns):
    mean = X[col].quantile(0.95)
    mean_list.append(mean)
    
# print num of features that mean < 1 
print(f"Num of features that mean < 1: {len([i for i in mean_list if i < 1])}")

# remove the features that mean < 1
# [i for i in mean_list if i >= 1]
X = X[[col for col, mean in zip(X.columns, mean_list) if mean >= 1]]

# Remove constant features
from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=0)
sel.fit(X)

sum(sel.get_support())

X = X[X.columns[sel.get_support()]]

print(f"Data shape after VarianceThreshold: {X.shape}")


# Remove quasi-constant features 
sel = VarianceThreshold(threshold=0.05)
sel.fit(X)

sum(sel.get_support())

X = X[X.columns[sel.get_support()]]
print(f"Data shape after quasi-constant features: {X.shape}")

# calculate the correlation with the target
corr = X.corrwith(data[target])
corr = corr.abs().sort_values(ascending=False)

# select features with correlation greater than 0.05
X = X[corr[corr > 0.03].index]

print(f"Data shape after correlation: {X.shape}")


# add clinical feature
# check clinical feature exist or not 
# if not, add it
for col in clinical_feature:
    if col not in X.columns:
        X[col] = data_clinical[col]
        
data = pd.concat([X, data[target]], axis=1)

# convert all columns to float except target 
X = data.drop([target], axis=1)
X = X.astype(float)
data = pd.concat([X, data[target]], axis=1)

print(f"Data shape after adding clinical feature: {data.shape}")

if n != 0:
    data_val = data_val.drop(col_list_drop, axis=1)
    data_val = data_val[data.columns]
    data_val = data_val.astype(float)

    print(f"Data validation shape: {data_val.shape}")
    

data_val = data_val[data.columns]
data_val = data_val.astype(float)
data = data.astype(float)

if not os.path.exists(os.path.join(root_dir, 'Academic/A01_RIF/data')):

    data.to_csv(os.path.join(root_dir, 'Academic/A01_RIF/data', f'{v}.csv'), index=False)
    data_val.to_csv(os.path.join(root_dir, 'Academic/A01_RIF/data', f'{v}_val.csv'), index=False)


    # data_register 
    ## Academic/A01_RIF/data/data_info.json
    with open(os.path.join(root_dir, 'Academic/A01_RIF/data/data_info.json'), 'r') as f:
        data_info = json.load(f)

    data_info[version] = {}
    data_info[version]['data_path'] = os.path.join(root_dir, 'Academic/A01_RIF/data', f'{v}.csv')
    data_info[version]['target'] = target
    data_info[version]['appendix'] = description
    data_info[version]['shape'] = np.shape(data)
    data_info[version]['val_exist'] = True if n != 0 else False
    data_info[version]['val_shape'] = np.shape(data_val) if n != 0 else None
    data_info[version]['data_val_path'] = os.path.join(root_dir, 'Academic/A01_RIF/data', f'{v}_val.csv')

    # save data_info
    with open(os.path.join(root_dir, 'Academic/A01_RIF/data/data_info.json'), 'w') as f:
        json.dump(data_info, f, indent=4)
        
else:
    print("Data already exists")

#######################################
# test
if __name__ == '__main__':
    # example 
    # os.system(f"python Academic/A01_RIF/method/data_process.py --data_version RIF_V2 --exp_version v4 --num_val 25 --target label_1 --description 'n400 val 50 随机挑选 label_1'")
    pass