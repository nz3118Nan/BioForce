#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   evaluation_plot.py
@Time    :   2024/09/04 12:23:59
@Author  :   Nan Zhou
@Version :   0.3
@Desc    :   None
'''
######################################################################
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
from Academic.A01_RIF.utils.func import *

######################################################################
# set a argument parser name data_version, need to be string
parser = ArgumentParser()
parser.add_argument("--exp_version", type=str, default='v1', help="exp version")    

######################################################################

# get data_version
args = parser.parse_args()

v = args.exp_version

version = v.upper()

# load the log
with open(f'Academic/A01_RIF/result/exp_log_{version}.json', 'r') as f:
    exp_log = json.load(f)
# num of experiment model 
num_exp = len(exp_log['model_list'][:6])    
model_short_list = exp_log['model_short_name'][:6]
model_list = exp_log['model_list'][:6]

# load the data
with open(os.path.join(root_dir, 'Academic/A01_RIF/data/data_info.json'), 'r') as f:
    data_info = json.load(f)

data_path = data_info[version]['data_path']
data = pd.read_csv(data_path)
target = data_info[version]['target'] 
shape = data_info[version]['shape'] 

clinical_feature = ['Age_at_embryo_transfer', 'BMI', 'AMH','num_of_embryo', 'embryo_evaluation_value']

# check existence of clinical feature
for feature in clinical_feature:
    if feature not in data.columns:
        print(f"Missing feature: {feature}")
        print(f"Available feature: {data.columns}")

# remove list 
remove_list = []

data = data.drop(columns=remove_list)

# check data_val exist
if data_info[version]['val_exist']:
    data_val_path = data_info[version]['data_val_path']
    data_val = pd.read_csv(data_val_path)
    val_shape = data_info[version]['val_shape'] 
    
    # check existence of clinical feature
    for feature in clinical_feature:
        if feature not in data_val.columns:
            print(f"Missing feature: {feature}")
            print(f"Available feature: {data_val.columns}")
    
    # remove list 
    data_val = data_val.drop(columns=remove_list)
    print(f"Validation set shape: {val_shape}")
print(f"Data shape: {shape}")
col_name = list(data.columns)


# importance analysis 
# overlap part of the importance analysis in the top 200 features in different models, at least 2 models have the feature in top 200 features

# importance_analysis_0_common = {} 
# importance_analysis_1_common = {}
importance_analysis_avg_common = {}

model_list = ["Ridge", "Lasso"]

num_exp = 50
for model in model_list:
    # importance_analysis_0_common[model] = {}
    # importance_analysis_1_common[model] = {}
    importance_analysis_avg_common[model] = {}
    
    for col in col_name:
        # importance_analysis_0_common[model][col] = {}
        # importance_analysis_0_common[model][col]['rank'] = []
        # importance_analysis_0_common[model][col]['value'] = []
        
        # importance_analysis_1_common[model][col] = {}
        # importance_analysis_1_common[model][col]['rank'] = []
        # importance_analysis_1_common[model][col]['value'] = []
        
        importance_analysis_avg_common[model][col] = {}
        importance_analysis_avg_common[model][col]['rank'] = []
        importance_analysis_avg_common[model][col]['value'] = []
        
for i in tqdm(range(num_exp)):
    for model in model_list:
        # for j, tmp in enumerate(exp_log[model]['importance_dic'][str(i)]['0']):
        #     importance_analysis_0_common[model][tmp[0]]['rank'].append(j)
        #     importance_analysis_0_common[model][tmp[0]]['value'].append(tmp[1])

        # for j, tmp in enumerate(exp_log[model]['importance_dic'][str(i)]['1']):
        #     importance_analysis_1_common[model][tmp[0]]['rank'].append(j)
        #     importance_analysis_1_common[model][tmp[0]]['value'].append(tmp[1])

        for j, tmp in enumerate(exp_log[model]['importance_dic'][str(i)]['avg']):
            importance_analysis_avg_common[model][tmp[0]]['rank'].append(j)
            importance_analysis_avg_common[model][tmp[0]]['value'].append(tmp[1])
            
# calculate the average rank and value
for model in model_list:
    for col in col_name:
        # importance_analysis_0_common[model][col]['rank_avg'] = np.mean(importance_analysis_0_common[model][col]['rank'])
        # importance_analysis_0_common[model][col]['value_avg'] = np.mean(importance_analysis_0_common[model][col]['value'])

        # importance_analysis_1_common[model][col]['rank_avg'] = np.mean(importance_analysis_1_common[model][col]['rank'])
        # importance_analysis_1_common[model][col]['value_avg'] = np.mean(importance_analysis_1_common[model][col]['value'])
        
        rank_list = importance_analysis_avg_common[model][col]['rank']
        value_list = importance_analysis_avg_common[model][col]['value']
        
        # sort the rank 
        rank_list = sorted(rank_list)
        value_list = sorted(value_list)
        
        # get rid of the highest and lowest 10% value
        rank_list = rank_list[int(len(rank_list) * 0.1): int(len(rank_list) * 0.9)]
        value_list = value_list[int(len(value_list) * 0.1): int(len(value_list) * 0.9)]

        importance_analysis_avg_common[model][col]['rank_avg'] = np.mean(importance_analysis_avg_common[model][col]['rank'])
        importance_analysis_avg_common[model][col]['value_avg'] = np.mean(importance_analysis_avg_common[model][col]['value'])


# num of experiment model 
num_exp = len(exp_log['model_list'][:6])    
model_short_list = exp_log['model_short_name'][:6]
model_list = exp_log['model_list'][:6]

# 2 * 2 subplot
plt.figure(figsize=(20, 20))
plt.subplot(3, 3, 1)
# plot 1 bar plot the all metrics for all models, using table 
## different color for different model
## metrics: BA, MCC, F1, Sen, Spe
## metrics as x-axis, value as y-axis
## same metrics no overlap, no gap between them 
## different model with gap between them

x_axis = ['BA', 'MCC', 'F1', 'Sen', 'Spe']
x_axis_loc = np.arange(len(x_axis))

width = 1 / (num_exp + 1)

# more colorful set 
from itertools import cycle
color = cycle(plt.cm.tab20(np.linspace(0, 1, num_exp)))

ba_list = []
mcc_list = []
f1_list = []
sen_list = []
spe_list = []

for i in range(num_exp):
    model = model_list[i]
    print(model)
    auc_dic = exp_log[model]['auc_dic'].copy()
    
    remove_key = []
    keys = list(auc_dic.keys())
    for key in keys:
        if type(key) != int:
            # convert to int
            auc_dic[int(key)] = auc_dic[key]
            # delete the original key
            remove_key.append(key)
            
    for key in remove_key:
        auc_dic.pop(key)
        
    ba_list_tmp, mcc_list_tmp, f1_list_tmp, sen_list_tmp, spe_list_tmp = auc_plot(auc_dic, data = 'avg', metric = True)
    
    ba = np.mean(ba_list_tmp)
    mcc = np.mean(mcc_list_tmp)
    f1 = np.mean(f1_list_tmp)
    sen = np.mean(sen_list_tmp)
    spe = np.mean(spe_list_tmp)
    
    ba_list.append(ba)
    mcc_list.append(mcc)
    f1_list.append(f1)
    sen_list.append(sen)
    spe_list.append(spe)

    plt.bar(x_axis_loc + i * width, [ba, mcc, f1, sen, spe], width=width, label=model_short_list[i], color=next(color))
    
# add the value on the top of the bar if the value is highest metric value in that metric
for i, value in enumerate([ba_list, mcc_list, f1_list, sen_list, spe_list]):
    max_value = max(value)
    for j, value in enumerate(value):
        if value == max_value:
            # add a smaill arrow to indicate the highest value
            plt.annotate('↑', (i + j * width, value), ha='center', va='bottom') 
            # add the value on the top of the bar, but move it up a little bit
            plt.text(i + j * width, value + 0.03, round(value, 2), ha='center', va='bottom')
    
plt.xticks(x_axis_loc + width * (num_exp - 1) / 2, x_axis)

plt.legend(loc='lower right')

plt.title('Model Test Performance')

# limit the y-axis to 1.2 
plt.ylim(0, 1.1)

# grid 
# plt.grid()

# remove the top and right line
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# change y-axis to 0.05 interval
plt.yticks(np.arange(0, 1., 0.05))

plt.subplot(3, 3, 2)
# plot 2 auc plot for all models 
## different color for different model
## plot the average line for each model

for i in range(num_exp):
    model = model_list[i]
    fpr_avg = [0] * 20
    tpr_avg = [0] * 20
    auc_avg_list = []
    for j in range(50):
        fpr, tpr, _ = roc_curve(exp_log[model]['auc_dic'][str(j)]['avg']['y_test'], exp_log[model]['auc_dic'][str(j)]['avg']['probs'])
        fpr = list(fpr)
        tpr = list(tpr)
        ## proprotionally extend the line to 20 points
        fpr = [fpr[int(i / 20 * len(fpr))] for i in range(20)]
        tpr = [tpr[int(i / 20 * len(tpr))] for i in range(20)]
        tpr_avg = [x + y for x, y in zip(tpr_avg, tpr)]
        fpr_avg = [x + y for x, y in zip(fpr_avg, fpr)]
        
        auc_avg_list.append(roc_auc_score(exp_log[model]['auc_dic'][str(j)]['avg']['y_test'], exp_log[model]['auc_dic'][str(j)]['avg']['probs']))
        
    # average line
    tpr_avg = [x / 50 for x in tpr_avg]
    fpr_avg = [x / 50 for x in fpr_avg]
    
    # calculate auc for average line using tpr_avg_1 and fpr_avg_1
    auc_avg = np.mean(auc_avg_list)
    
    # plot the average line, dashed line
    plt.plot(fpr_avg, tpr_avg, label=f'Model {model}, AUC: {round(auc_avg, 2)}', linestyle='dashed')  
    
    # shade the area between the average line and the x-axis, with grey color
    plt.fill_between(fpr_avg, tpr_avg, 0, alpha=0.1, color='grey')
    

# add grid 
plt.grid()

# remove the top and right line
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
    
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Model (Test)')

# large font size for the legend
plt.legend(fontsize='large')


plt.subplot(3, 3, 3)
# plot 1 bar plot the all metrics for all models, using table 
## different color for different model
## metrics: BA, MCC, F1, Sen, Spe
## metrics as x-axis, value as y-axis
## same metrics no overlap, no gap between them 
## different model with gap between them

x_axis = ['BA', 'MCC', 'F1', 'Sen', 'Spe']
x_axis_loc = np.arange(len(x_axis))

width = 1 / (num_exp + 1)

# more colorful set 
from itertools import cycle
color = cycle(plt.cm.tab20(np.linspace(0, 1, num_exp)))

ba_list = []
mcc_list = []
f1_list = []
sen_list = []
spe_list = []

for i in range(num_exp):
    model = model_list[i]
    
    auc_dic = exp_log[model]['auc_dic'].copy()
    
    remove_key = []
    keys = list(auc_dic.keys())
    for key in keys:
        if type(key) != int:
            # convert to int
            auc_dic[int(key)] = auc_dic[key]
            # delete the original key
            remove_key.append(key)
            
    for key in remove_key:
        auc_dic.pop(key)
        
    ba_list_tmp, mcc_list_tmp, f1_list_tmp, sen_list_tmp, spe_list_tmp = auc_plot(auc_dic, data = 'avg', metric = True, index_independent= 'spe', weight = 0.5)
    
    ba = np.mean(ba_list_tmp)
    mcc = np.mean(mcc_list_tmp)
    f1 = np.mean(f1_list_tmp)
    sen = np.mean(sen_list_tmp)
    spe = np.mean(spe_list_tmp)
    
    ba_list.append(ba)
    mcc_list.append(mcc)
    f1_list.append(f1)
    sen_list.append(sen)
    spe_list.append(spe)

    plt.bar(x_axis_loc + i * width, [ba, mcc, f1, sen, spe], width=width, label=model_short_list[i], color=next(color))
    
# add the value on the top of the bar if the value is highest metric value in that metric
for i, value in enumerate([ba_list, mcc_list, f1_list, sen_list, spe_list]):
    max_value = max(value)
    for j, value in enumerate(value):
        if value == max_value:
            # add a smaill arrow to indicate the highest value
            plt.annotate('↑', (i + j * width, value), ha='center', va='bottom') 
            # add the value on the top of the bar, but move it up a little bit
            plt.text(i + j * width, value + 0.03, round(value, 2), ha='center', va='bottom')
            
            if i == 4:
                # add a arrow to indicate the improvement ⇗, enlarge the font size
                plt.annotate('⇗', (i + j * width, value+0.05), ha='center', va='bottom', fontsize=40, color='red')
                
            if i == 3:
                # add a arrow to indicate the decrease ⇘, enlarge the font size
                plt.annotate('⇘', (i + j * width, value+0.1), ha='center', va='bottom', fontsize=40, color='green')
                
            if i == 2:
                # add a arrow to indicate the decrease ⇘, enlarge the font size
                plt.annotate('⇘', (i + j * width, value+0.1), ha='center', va='bottom', fontsize=40, color='green')
    
plt.xticks(x_axis_loc + width * (num_exp - 1) / 2, x_axis)

plt.legend(loc='lower right')

plt.title('Model Test Performance')

# limit the y-axis to 1.2 
plt.ylim(0, 1.1)

# remove the top and right line
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# change y-axis to 0.05 interval
plt.yticks(np.arange(0, 1., 0.05))

plt.subplot(3, 3, 4)
# plot 1 bar plot the all metrics for all models, using table val
## different color for different model
## metrics: BA, MCC, F1, Sen, Spe
## metrics as x-axis, value as y-axis
## same metrics no overlap, no gap between them 
## different model with gap between them

x_axis = ['BA', 'MCC', 'F1', 'Sen', 'Spe']
x_axis_loc = np.arange(len(x_axis))

width = 1 / (num_exp + 1)

# more colorful set 
from itertools import cycle
color = cycle(plt.cm.tab20(np.linspace(0, 1, num_exp)))

ba_list = []
mcc_list = []
f1_list = []
sen_list = []
spe_list = []

num_exp_val = len(exp_log[model_list[0]]['ba_val_avg'])

for i in range(num_exp):
    model = model_list[i]
    auc_dic_val = exp_log[model]['auc_dic_val'].copy()
    
    remove_key = []
    keys = list(auc_dic_val.keys())
    for key in keys:
        if type(key) != int:
            auc_dic_val[int(key)] = auc_dic_val[key]
            remove_key.append(key)
            
    for key in remove_key:
        auc_dic_val.pop(key)
    
    ba_list_tmp, mcc_list_tmp, f1_list_tmp, sen_list_tmp, spe_list_tmp = auc_plot(auc_dic_val, data = 'avg', metric = True)
    
    ba = np.mean(ba_list_tmp)
    mcc = np.mean(mcc_list_tmp)
    f1 = np.mean(f1_list_tmp)
    sen = np.mean(sen_list_tmp)
    spe = np.mean(spe_list_tmp)
    
    ba_list.append(ba)
    mcc_list.append(mcc)
    f1_list.append(f1)
    sen_list.append(sen)
    spe_list.append(spe)

    plt.bar(x_axis_loc + i * width, [ba, mcc, f1, sen, spe], width=width, label=model_short_list[i], color=next(color))
    

# add the value on the top of the bar if the value is highest metric value in that metric
for i, value in enumerate([ba_list, mcc_list, f1_list, sen_list, spe_list]):
    max_value = max(value)
    for j, value in enumerate(value):
        if value == max_value:
            # add a smaill arrow to indicate the highest value
            plt.annotate('↑', (i + j * width, value), ha='center', va='bottom') 
            # add the value on the top of the bar, but move it up a little bit
            plt.text(i + j * width, value + 0.03, round(value, 2), ha='center', va='bottom')
            
plt.xticks(x_axis_loc + width * (num_exp - 1) / 2, x_axis)

plt.legend(loc='lower right')

plt.title('Model Val Performance')

# limit the y-axis to 1.2 
plt.ylim(0, 1.1)

# remove the top and right line
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# change y-axis to 0.05 interval
plt.yticks(np.arange(0, 1., 0.05))

plt.subplot(3, 3, 5)
# plot 2 auc plot for all models  val 
## different color for different model
## plot the average line for each model

for i in range(num_exp):
    model = model_list[i]
    fpr_avg = [0] * 20
    tpr_avg = [0] * 20
    auc_avg_list = []
    num_exp_val = len(exp_log[model]['auc_dic_val'])
    for j in range(num_exp_val):
        fpr, tpr, _ = roc_curve(exp_log[model]['auc_dic_val'][str(j)]['avg']['y_test'], exp_log[model]['auc_dic_val'][str(j)]['avg']['probs'])
        fpr = list(fpr)
        tpr = list(tpr)
        ## proprotionally extend the line to 20 points
        fpr = [fpr[int(i / 20 * len(fpr))] for i in range(20)]
        tpr = [tpr[int(i / 20 * len(tpr))] for i in range(20)]
        tpr_avg = [x + y for x, y in zip(tpr_avg, tpr)]
        fpr_avg = [x + y for x, y in zip(fpr_avg, fpr)]
        
        auc_avg_list.append(roc_auc_score(exp_log[model]['auc_dic_val'][str(j)]['avg']['y_test'], exp_log[model]['auc_dic_val'][str(j)]['avg']['probs']))
        
    # average line
    tpr_avg = [x / 50 for x in tpr_avg]
    fpr_avg = [x / 50 for x in fpr_avg]
    
    # calculate auc for average line using tpr_avg_1 and fpr_avg_1
    auc_avg = np.mean(auc_avg_list)
    
    # plot the average line, dashed line
    plt.plot(fpr_avg, tpr_avg, label=f'Model {model}, AUC: {round(auc_avg, 2)}', linestyle='dashed')  
    
    # shade the area between the average line and the x-axis, with grey color
    plt.fill_between(fpr_avg, tpr_avg, 0, alpha=0.1, color='grey')
    
# add grid 
plt.grid()

# remove the
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Model (Val)')
plt.legend(fontsize='large')

plt.subplot(3, 3, 6)
# plot 1 bar plot the all metrics for all models, using table val
## different color for different model
## metrics: BA, MCC, F1, Sen, Spe
## metrics as x-axis, value as y-axis
## same metrics no overlap, no gap between them 
## different model with gap between them

x_axis = ['BA', 'MCC', 'F1', 'Sen', 'Spe']
x_axis_loc = np.arange(len(x_axis))

width = 1 / (num_exp + 1)

# more colorful set 
from itertools import cycle
color = cycle(plt.cm.tab20(np.linspace(0, 1, num_exp)))

ba_list = []
mcc_list = []
f1_list = []
sen_list = []
spe_list = []

num_exp_val = len(exp_log[model_list[0]]['ba_val_avg'])

for i in range(num_exp):
    model = model_list[i]
    auc_dic_val = exp_log[model]['auc_dic_val'].copy()
    
    remove_key = []
    keys = list(auc_dic_val.keys())
    for key in keys:
        if type(key) != int:
            auc_dic_val[int(key)] = auc_dic_val[key]
            remove_key.append(key)
            
    for key in remove_key:
        auc_dic_val.pop(key)
    
    ba_list_tmp, mcc_list_tmp, f1_list_tmp, sen_list_tmp, spe_list_tmp = auc_plot(auc_dic_val, data = 'avg', metric = True, index_independent= 'spe', weight = 0.3)
    
    ba = np.mean(ba_list_tmp)
    mcc = np.mean(mcc_list_tmp)
    f1 = np.mean(f1_list_tmp)
    sen = np.mean(sen_list_tmp)
    spe = np.mean(spe_list_tmp)
    
    ba_list.append(ba)
    mcc_list.append(mcc)
    f1_list.append(f1)
    sen_list.append(sen)
    spe_list.append(spe)

    plt.bar(x_axis_loc + i * width, [ba, mcc, f1, sen, spe], width=width, label=model_short_list[i], color=next(color))
    

# add the value on the top of the bar if the value is highest metric value in that metric
for i, value in enumerate([ba_list, mcc_list, f1_list, sen_list, spe_list]):
    max_value = max(value)
    for j, value in enumerate(value):
        if value == max_value:
            # add a smaill arrow to indicate the highest value
            plt.annotate('↑', (i + j * width, value), ha='center', va='bottom') 
            # add the value on the top of the bar, but move it up a little bit
            plt.text(i + j * width, value + 0.03, round(value, 2), ha='center', va='bottom')
            
            if i == 4:
                # add a arrow to indicate the improvement ⇗, enlarge the font size
                plt.annotate('⇗', (i + j * width, value+0.1), ha='center', va='bottom', fontsize=40, color='red')
                
            if i == 3:
                # add a arrow to indicate the decrease ⇘, enlarge the font size
                plt.annotate('⇘', (i + j * width, value+0.1), ha='center', va='bottom', fontsize=40, color='green')
                
            if i == 2:
                # add a arrow to indicate the decrease ⇘, enlarge the font size
                plt.annotate('⇘', (i + j * width, value+0.1), ha='center', va='bottom', fontsize=40, color='green')
            
            
plt.xticks(x_axis_loc + width * (num_exp - 1) / 2, x_axis)

plt.legend(loc='lower right')

plt.title('Model Val Performance')

# limit the y-axis to 1.2 
plt.ylim(0, 1.1)

# remove the top and right line
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# change y-axis to 0.05 interval
plt.yticks(np.arange(0, 1., 0.05))
    
plt.subplot(3, 3, 7)
# plot 3 importance analysis rank for top overlap features
# select the top 20 features by rank
n = 20
color = cycle(plt.cm.tab20(np.linspace(0, 1, num_exp)))

# width thin 
plt.barh([x for x in feature_list_top[:n]], [importance_analysis_avg_common['LogisticRegression'][x]['rank_avg'] for x in feature_list_top[:n]], color= 'orange', height=0.2)

# remove the top and right line
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# empty x-axis label
plt.xticks([])
plt.xlabel('Importance Relative Rank')

# change y-axis position 
plt.yticks(np.arange(n), feature_list_top[:n])

# add the horizontal line for each feature, indicating the rank of the feature in different models 
for i in range(n):
    plt.axhline(y=i, color='grey', linestyle='dotted')

plt.title(f'Model Importance Analysis by Rank for Top {n} Features')

plt.subplot(3, 3, 8)
# violin plot for the top 20 features by value, moving down, matching the height of the bar plot
plt.violinplot([importance_analysis_avg_common['LogisticRegression'][x]['value'] for x in feature_list_top[:n]], vert=False, positions=np.arange(n), showmeans=False, showmedians=True)

# red point for the average value of the feature
plt.scatter([importance_analysis_avg_common['LogisticRegression'][x]['value_avg'] for x in feature_list_top[:n]], np.arange(n), color='red', label='average value')

# vertical line at 0
plt.axvline(x=0, color='grey', linestyle='--')

# change y-axis position 
plt.yticks(np.arange(n), [])

# remove the top and right line, left line
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)

plt.legend()
plt.xlabel('Importance Value')

plt.title(f'Model Importance Analysis by Value for Top {n} Features')


# add dashed horizontal line for each feature, indicating the rank of the feature in different models 
# between the violin plot and the bar plot
for i in range(n):
    plt.axhline(y=i, color='grey', linestyle='dotted')

# save the figure
plt.savefig(f'Academic/A01_RIF/result/{version}_model_performance.png')

######################################################################
# test
if __name__ == '__main__':
    # example 
    # os.system(f"python Academic/A01_RIF/method/evaluation_plot.py --exp_version v1")
    pass