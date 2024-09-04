#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   result_generation.py
@Time    :   2024/09/04 11:36:05
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

with open(os.path.join(root_dir, 'Academic/A01_RIF/data/data_info.json'), 'r') as f:
    data_info = json.load(f)
    
data_path = data_info[version]['data_path']
data = pd.read_csv(data_path)

if data_info[version]['val_exist']:
    data_val_path = data_info[version]['data_val_path']
    data_val = pd.read_csv(data_val_path)
    

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


X = data.drop(columns=[target])
y = data[target]


# PCA plot
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X)


X = data.drop(columns=[target])
y = data[target]

if data_info[version]['val_exist']:
    X_val = data_val.drop(columns=[target])
    y_val = data_val[target]
    
data_train_copy = data.copy()
data_val_copy = data_val.copy()

# PCA 999% on X
pca = PCA(n_components=0.999, svd_solver='full')
X_pca = pca.fit_transform(X)
# array to dataframe with columns name 
X_pca = pd.DataFrame(X_pca, columns=[f"component_{i}" for i in range(X_pca.shape[1])])

X_feature_importance = pca.components_

if data_info[version]['val_exist']:
    X_val_pca = pca.transform(X_val)
    X_val_pca = pd.DataFrame(X_val_pca, columns=[f"component_{i}" for i in range(X_val_pca.shape[1])])
    
print(f"X_pca shape: {X_pca.shape}")
print(f"X_val_pca shape: {X_val_pca.shape}")


# reset index
X.reset_index(drop=True, inplace=True)
y.reset_index(drop=True, inplace=True)

if data_info[version]['val_exist']:
    X_val.reset_index(drop=True, inplace=True)
    y_val.reset_index(drop=True, inplace=True)
    
    

# updata X and data
X = X_pca
data = pd.concat([X, y], axis=1)

if data_info[version]['val_exist']:
    X_val = X_val_pca
    data_val = pd.concat([X_val, y_val], axis=1)
    
# function to random split data 
from sklearn.model_selection import train_test_split
def random_split_data(data, target, random_seed):
    X = data.drop(columns=[target])
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_seed)
    return X_train, X_test, y_train, y_test

X = data.drop(columns=[target])
y = data[target]

if data_info[version]['val_exist']:
    X_val = data_val.drop(columns=[target])
    y_val = data_val[target]
    

# load the data
import json 

# if exp_log_version.json do not exist, create it
exp_log_path = f'Academic/A01_RIF//result/exp_log_{version}.json'

if not os.path.exists(exp_log_path):
    with open(exp_log_path, 'w') as f:
        json.dump({}, f)
        
# load the data
with open(exp_log_path, 'r') as f:
    exp_log = json.load(f)
    
exp_model = ['LogisticRegression', "SVM", "Ridge", "Lasso",  "NaiveBayes", "NeuralNetwork", "XGBoost", "LightGBM", "CatBoost"]
exp_log['version'] = version
exp_short_name = ['LogReg', 'SVM', 'Ridge', 'Lasso', 'NB', 'NN', 'XGB', 'LGB', 'CB']
exp_log['model_short_name'] = exp_short_name
exp_log['model_list'] = exp_model
for i in exp_model:
    if i not in exp_log.keys():
        exp_log[i] = {} 
        
# function to train model and evaluate model use logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
def train_evaluate_model(data, target, random_seed, data_val=None):
    X_train, X_test, y_train, y_test = random_split_data(data, target, random_seed)
    
    # # grid search for hyperparameter C in logistic regression
    # from sklearn.model_selection import GridSearchCV
    # param_grid = {'C': [0.05, 0.1, 1, 10, 20]}
    
    # best_hyperparams = GridSearchCV(LogisticRegression(random_state=random_seed, class_weight='balanced'), param_grid, cv=5, scoring='f1_weighted', n_jobs=-1, verbose= 4).fit(X_train, y_train).best_params_
    # print(f"Best hyperparams: {best_hyperparams}")
    
    model = LogisticRegression(random_state=random_seed, class_weight='balanced')
    model.fit(X_train, y_train)
    
    # adjust the threshold to avoid low specificity
    threshold = 0.33
    
    y_pred = model.predict(X_test)
    prod = model.predict_proba(X_test)[:, 1]
    y_pred = [1 if i > threshold else 0 for i in prod]
    
    ba, mcc, f1, cm, sen, spe = evaluation(y_test, y_pred)
    
    # also output the probs for roc
    probs = model.predict_proba(X_test)[:, 1]
    
    auc_dic = {}
    auc_dic['probs'] = probs
    auc_dic['y_test'] = y_test
    auc_dic['y_pred'] = y_pred
    auc_dic['X_test'] = [list(X_test.iloc[i]) for i in range(len(X_test))]
    
    # importance 
    result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=random_seed)
    importance = result.importances
    importance = np.mean(importance, axis=1)
    if data_val is not None:
        X_val = data_val.drop(columns=[target])
        y_val = data_val[target]
        y_pred_val = model.predict(X_val)
        prod_val = model.predict_proba(X_val)[:, 1]
        y_pred_val = [1 if i > threshold else 0 for i in prod_val]
        
        auc_val_dic = {}
        auc_val_dic['probs'] = prod_val
        auc_val_dic['y_test'] = y_val
        auc_val_dic['y_pred'] = y_pred_val
        
        ba_val, mcc_val, f1_val, cm_val, sen_val, spe_val,  = evaluation(y_val, y_pred_val)
        return {'test': [ba, mcc, f1, cm, sen, spe, auc_dic, importance], 'val': [ba_val, mcc_val, f1_val, cm_val, sen_val, spe_val, auc_val_dic]}
    
    return ba, mcc, f1, cm, sen, spe, auc_dic,importance

# num of experiment 
num_exp = 50
model_name = 'LogisticRegression'
threshold = 0
save = True

if True:
    ba_list = []
    mcc_list = []
    f1_list = []
    sen_list = []
    spe_list = []
    importance_dic = {}
    auc_dic = {}

    ba_val_list = []
    mcc_val_list = []
    f1_val_list = []
    sen_val_list = []
    spe_val_list = []
    auc_dic_val = {}

# check existence of result in exp_log
if 'auc_dic' not in exp_log[model_name].keys():
    # run the experiment
    for i in tqdm(range(num_exp)):
        random_seed = np.random.randint(0, 100)
        dic = train_evaluate_model(data, target, random_seed, data_val)
        ba_avg, mcc_avg, f1_avg, cm_avg, sen_avg, spe_avg, auc_dic_avg, importance_avg = dic['test']
        ba_val, mcc_val, f1_val, cm_val, sen_val, spe_val, auc_dic_val_avg = dic['val']
        ba_list.append(ba_avg)
        mcc_list.append(mcc_avg)
        f1_list.append(f1_avg)
        sen_list.append(sen_avg)
        spe_list.append(spe_avg)
        
        importance_avg = list(importance_avg.T @ X_feature_importance)
        importance_avg = [(col_name[i], importance_avg[i]) for i in range(len(importance_avg))]
        importance_avg = sorted(importance_avg, key=lambda x: abs(x[1]), reverse=True)
        
        auc_dic[i] = {}
        auc_dic[i]['avg'] = {}
        auc_dic[i]['avg'] = auc_dic_avg
        
        importance_dic[i] = {}
        importance_dic[i]['avg'] = {}
        importance_dic[i]['avg'] = importance_avg
        
        ba_val_list.append(ba_val)
        mcc_val_list.append(mcc_val)
        f1_val_list.append(f1_val)
        sen_val_list.append(sen_val)
        spe_val_list.append(spe_val)
        
        auc_dic_val[i] = {}
        auc_dic_val[i]['avg'] = {}
        auc_dic_val[i]['avg'] = auc_dic_val_avg
        
    # process the result
    if True:
        remove_key = []
        keys = list(importance_dic.keys())
        for i in keys:
            if type(i) != int:
                # convert to int
                importance_dic[int(i)] = importance_dic[i]
                # delete the original key
                remove_key.append(i)
                
        for i in remove_key:
            importance_dic.pop(i)

        # change all type of root of the value in auc_dic to list 
        for i in auc_dic:
            for j in auc_dic[i]:
                for k in auc_dic[i][j]:
                    auc_dic[i][j][k] = list(auc_dic[i][j][k]) 
            
        for i in auc_dic_val:
            for j in auc_dic_val[i]:
                for k in auc_dic_val[i][j]:
                    auc_dic_val[i][j][k] = list(auc_dic_val[i][j][k])
    
    # filter 
    if True:       
        average_auc_score_list = auc_plot(auc_dic, data = 'avg', cal = True)
        auc_score_list = [1 if i > threshold else 0 for i in average_auc_score_list]

        # only keep the trial that is larger than threshold for val 
        ba_val_list = [ba_val_list[i] for i in range(len(auc_score_list)) if auc_score_list[i] == 1]
        mcc_val_list = [mcc_val_list[i] for i in range(len(auc_score_list)) if auc_score_list[i] == 1] 
        f1_val_list = [f1_val_list[i] for i in range(len(auc_score_list)) if auc_score_list[i] == 1]
        sen_val_list = [sen_val_list[i] for i in range(len(auc_score_list)) if auc_score_list[i] == 1]
        spe_val_list = [spe_val_list[i] for i in range(len(auc_score_list)) if auc_score_list[i] == 1]

        auc_dic_val = {k: auc_dic_val[k] for k in range(len(auc_score_list)) if auc_score_list[k] == 1}

        # reorder the auc_dic_val
        new_key_list = list(auc_dic_val.keys())
        new_aic_dic_val = {}
        for i in range(len(new_key_list)):
            new_aic_dic_val[i] = auc_dic_val[new_key_list[i]]
            auc_dic_val.pop(new_key_list[i])

        # update the auc_dic_val
        auc_dic_val = new_aic_dic_val

        print(f"Number of trial that is larger than threshold: {len(auc_score_list)}")
    
    if save:
        # exp_log 
        exp_log[model_name]['ba_avg'] = list(ba_list)
        exp_log[model_name]['mcc_avg'] = list(mcc_list)
        exp_log[model_name]['f1_avg'] = list(f1_list)
        exp_log[model_name]['sen_avg'] = list(sen_list)
        exp_log[model_name]['spe_avg'] = list(spe_list)
        exp_log[model_name]['auc_dic'] = auc_dic
        exp_log[model_name]['importance_dic'] = importance_dic

        exp_log[model_name]['ba_val_avg'] = list(ba_val_list)
        exp_log[model_name]['mcc_val_avg'] = list(mcc_val_list)
        exp_log[model_name]['f1_val_avg'] = list(f1_val_list)
        exp_log[model_name]['sen_val_avg'] = list(sen_val_list)
        exp_log[model_name]['spe_val_avg'] = list(spe_val_list)
        exp_log[model_name]['auc_dic_val'] = auc_dic_val

        # save the log
        with open(f'Academic/A01_RIF/result/exp_log_{version}.json', 'w') as f:
            json.dump(exp_log, f, indent=4)

# SVM 
from sklearn.svm import SVC

def train_evaluate_model_SVC(data, target, random_seed, data_val=None):
    X_train, X_test, y_train, y_test = random_split_data(data, target, random_seed)
    
    # hyperparameter tuning for SVM 
    hyperparameter = {'C': [1,5,10],   # 1 5 10 50 100
              'gamma': [0.00005,0.0001],  # 0.00001 0.00005 0.0001 0.0005
              'kernel': ['rbf']} 
    from sklearn.model_selection import GridSearchCV
    grid = GridSearchCV(SVC(), hyperparameter, refit=True, verbose=0)
    grid.fit(X_train, y_train)
    
    # best hyperparameter
    # print(grid.best_params_)
    
    model = SVC(C=grid.best_params_['C'], gamma=grid.best_params_['gamma'], kernel=grid.best_params_['kernel'], random_state=random_seed, class_weight='balanced')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # threshold = 0.5
    # prob = model.decision_function(X_test)
    # y_pred = [1 if i > threshold else 0 for i in prob]
    decision = model.decision_function(X_test)
    # convert decision to probability
    probs = (decision - min(decision)) / (max(decision) - min(decision))
    
    threshold = 0.55
    y_pred = [1 if i > threshold else 0 for i in probs]
    # print(probs)
    
    # importance 
    result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=random_seed)
    importance = result.importances
    importance = np.mean(importance, axis=1)
    
    ba, mcc, f1, cm, sen, spe = evaluation(y_test, y_pred)
    
    auc_dic = {}
    auc_dic['probs'] = probs
    auc_dic['y_test'] = y_test
    auc_dic['y_pred'] = y_pred
    auc_dic['X_test'] = [list(X_test.iloc[i]) for i in range(len(X_test))]
    
    if data_val is not None:
        X_val = data_val.drop(columns=[target])
        y_val = data_val[target]
        y_pred_val = model.predict(X_val)
        decision_val = model.decision_function(X_val)
        probs_val = (decision_val - min(decision_val)) / (max(decision_val) - min(decision_val))
        y_pred_val = [1 if i > threshold else 0 for i in probs_val]
        
        auc_val_dic = {}
        auc_val_dic['probs'] = probs_val
        auc_val_dic['y_test'] = y_val
        auc_val_dic['y_pred'] = y_pred_val
        
        ba_val, mcc_val, f1_val, cm_val, sen_val, spe_val = evaluation(y_val, y_pred_val)
        return {'test': [ba, mcc, f1, cm, sen, spe, auc_dic, grid.best_params_, importance], 'val': [ba_val, mcc_val, f1_val, cm_val, sen_val, spe_val, auc_val_dic]}
    
    
    return ba, mcc, f1, cm, sen, spe, auc_dic,grid.best_params_, importance

# num of experiment 
num_exp = 50
model_name = 'SVM'
threshold = 0
save = True

if True:
    ba_list = []
    mcc_list = []
    f1_list = []
    sen_list = []
    spe_list = []
    p_avg_list = []

    importance_dic = {}

    ba_val_list = []
    mcc_val_list = []
    f1_val_list = []
    sen_val_list = []
    spe_val_list = []

    auc_dic = {}
    auc_dic_val = {}
    

# check existence of result in exp_log
if 'auc_dic' not in exp_log[model_name].keys():
    # run the experiment
    for i in tqdm(range(num_exp)):
        random_seed = np.random.randint(0, 100)
        
        dic = train_evaluate_model_SVC(data, target, random_seed, data_val)
        ba_avg, mcc_avg, f1_avg, cm_avg, sen_avg, spe_avg, auc_dic_avg, p_avg, importance_avg = dic['test']
        ba_val, mcc_val, f1_val, cm_val, sen_val, spe_val, auc_dic_val_avg = dic['val']
        
        ba_list.append(ba_avg)
        mcc_list.append(mcc_avg)
        f1_list.append(f1_avg)
        sen_list.append(sen_avg)
        spe_list.append(spe_avg)
        p_avg_list.append(p_avg)
        
        importance_avg = list(importance_avg.T @ X_feature_importance)
        importance_avg = [(col_name[i], importance_avg[i]) for i in range(len(importance_avg))]
        importance_avg = sorted(importance_avg, key=lambda x: abs(x[1]), reverse=True)
        
        auc_dic[i] = {}
        auc_dic[i]['avg'] = {}
        auc_dic[i]['avg'] = auc_dic_avg
        
        importance_dic[i] = {}
        importance_dic[i]['avg'] = {}
        importance_dic[i]['avg'] = importance_avg
        
        ba_val_list.append(ba_val)
        mcc_val_list.append(mcc_val)
        f1_val_list.append(f1_val)
        sen_val_list.append(sen_val)
        spe_val_list.append(spe_val)
        
        auc_dic_val[i] = {}
        auc_dic_val[i]['avg'] = {}
        auc_dic_val[i]['avg'] = auc_dic_val_avg
        
    
    # process the result
    if True:
        remove_key = []
        keys = list(importance_dic.keys())
        for i in keys:
            if type(i) != int:
                # convert to int
                importance_dic[int(i)] = importance_dic[i]
                # delete the original key
                remove_key.append(i)
                
        for i in remove_key:
            importance_dic.pop(i)

        # change all type of root of the value in auc_dic to list 
        for i in auc_dic:
            for j in auc_dic[i]:
                for k in auc_dic[i][j]:
                    auc_dic[i][j][k] = list(auc_dic[i][j][k]) 
            
        for i in auc_dic_val:
            for j in auc_dic_val[i]:
                for k in auc_dic_val[i][j]:
                    auc_dic_val[i][j][k] = list(auc_dic_val[i][j][k])
    
    # filter 
    if True:       
        average_auc_score_list = auc_plot(auc_dic, data = 'avg', cal = True)
        auc_score_list = [1 if i > threshold else 0 for i in average_auc_score_list]

        # only keep the trial that is larger than threshold for val 
        ba_val_list = [ba_val_list[i] for i in range(len(auc_score_list)) if auc_score_list[i] == 1]
        mcc_val_list = [mcc_val_list[i] for i in range(len(auc_score_list)) if auc_score_list[i] == 1] 
        f1_val_list = [f1_val_list[i] for i in range(len(auc_score_list)) if auc_score_list[i] == 1]
        sen_val_list = [sen_val_list[i] for i in range(len(auc_score_list)) if auc_score_list[i] == 1]
        spe_val_list = [spe_val_list[i] for i in range(len(auc_score_list)) if auc_score_list[i] == 1]

        auc_dic_val = {k: auc_dic_val[k] for k in range(len(auc_score_list)) if auc_score_list[k] == 1}

        # reorder the auc_dic_val
        new_key_list = list(auc_dic_val.keys())
        new_aic_dic_val = {}
        for i in range(len(new_key_list)):
            new_aic_dic_val[i] = auc_dic_val[new_key_list[i]]
            auc_dic_val.pop(new_key_list[i])

        # update the auc_dic_val
        auc_dic_val = new_aic_dic_val

        print(f"Number of trial that is larger than threshold: {len(auc_score_list)}")
    
    if save:
        # exp_log 
        exp_log[model_name]['ba_avg'] = list(ba_list)
        exp_log[model_name]['mcc_avg'] = list(mcc_list)
        exp_log[model_name]['f1_avg'] = list(f1_list)
        exp_log[model_name]['sen_avg'] = list(sen_list)
        exp_log[model_name]['spe_avg'] = list(spe_list)
        exp_log[model_name]['auc_dic'] = auc_dic
        exp_log[model_name]['importance_dic'] = importance_dic
        exp_log[model_name]['p_avg'] = list(p_avg_list)

        exp_log[model_name]['ba_val_avg'] = list(ba_val_list)
        exp_log[model_name]['mcc_val_avg'] = list(mcc_val_list)
        exp_log[model_name]['f1_val_avg'] = list(f1_val_list)
        exp_log[model_name]['sen_val_avg'] = list(sen_val_list)
        exp_log[model_name]['spe_val_avg'] = list(spe_val_list)
        exp_log[model_name]['auc_dic_val'] = auc_dic_val

        # save the log
        with open(f'Academic/A01_RIF/result/exp_log_{version}.json', 'w') as f:
            json.dump(exp_log, f, indent=4)


# ridge regression
from sklearn.linear_model import Ridge

def train_evaluate_model_Ridge(data, target, random_seed, data_val = None):
    
    X_train, X_test, y_train, y_test = random_split_data(data, target, random_seed)
    
    # hyperparameter tuning for Ridge 
    hyperparameter = {'alpha': [0.1, 1, 10, 100, 1000]} 
    from sklearn.model_selection import GridSearchCV
    grid = GridSearchCV(Ridge(), hyperparameter, refit=True, verbose=0)
    grid.fit(X_train, y_train)
    
    # best hyperparameter
    # print(grid.best_params_)
    
    model = Ridge(alpha=grid.best_params_['alpha'], random_state=random_seed)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    probs = model.predict(X_test)
    
    threshold = 0.2
    y_pred = [1 if i > threshold else 0 for i in probs]
    # print(probs)
    
    ba, mcc, f1, cm, sen, spe = evaluation(y_test, y_pred)
    
    auc_dic = {}
    auc_dic['probs'] = probs
    auc_dic['y_test'] = y_test
    auc_dic['y_pred'] = y_pred
    auc_dic['X_test'] = [list(X_test.iloc[i]) for i in range(len(X_test))]
    
    # importance 
    importance = model.coef_
    
    if data_val is not None:
        X_val = data_val.drop(columns=[target])
        y_val = data_val[target]
        probs = model.predict(X_val)
        
        y_val_pred = [1 if i > threshold else 0 for i in probs]
        
        auc_dic_val = {}
        
        auc_dic_val['probs'] = probs
        auc_dic_val['y_test'] = y_val
        auc_dic_val['y_pred'] = y_val_pred

        ba_val, mcc_val, f1_val, cm_val, sen_val, spe_val = evaluation(y_val, y_val_pred)
        
        return {"test": [ba, mcc, f1, cm, sen, spe, auc_dic, grid.best_params_, importance], "val": [ba_val, mcc_val, f1_val, cm_val, sen_val, spe_val, auc_dic_val]}
    
    return ba, mcc, f1, cm, sen, spe, auc_dic, grid.best_params_, importance

# num of experiment 
num_exp = 50
model_name = 'Ridge'
threshold = 0
save = True

if True:
    ba_list = []
    mcc_list = []
    f1_list = []
    sen_list = []
    spe_list = []
    p_avg_list = []

    importance_dic = {}

    ba_val_list = []
    mcc_val_list = []
    f1_val_list = []
    sen_val_list = []
    spe_val_list = []

    auc_dic = {}
    auc_dic_val = {}
    

# check existence of result in exp_log
if 'auc_dic' not in exp_log[model_name].keys():
    # run the experiment
    for i in tqdm(range(num_exp)):
        random_seed = np.random.randint(0, 100)
        
        dic = train_evaluate_model_Ridge(data, target, random_seed, data_val)
        ba_avg, mcc_avg, f1_avg, cm_avg, sen_avg, spe_avg, auc_dic_avg, p_avg, importance_avg = dic['test']
        ba_val, mcc_val, f1_val, cm_val, sen_val, spe_val, auc_dic_val_avg = dic['val']
        
        ba_list.append(ba_avg)
        mcc_list.append(mcc_avg)
        f1_list.append(f1_avg)
        sen_list.append(sen_avg)
        spe_list.append(spe_avg)
        p_avg_list.append(p_avg)
        
        importance_avg = list(importance_avg.T @ X_feature_importance)
        importance_avg = [(col_name[i], importance_avg[i]) for i in range(len(importance_avg))]
        importance_avg = sorted(importance_avg, key=lambda x: abs(x[1]), reverse=True)
        
        auc_dic[i] = {}
        auc_dic[i]['avg'] = {}
        auc_dic[i]['avg'] = auc_dic_avg
        
        importance_dic[i] = {}
        importance_dic[i]['avg'] = {}
        importance_dic[i]['avg'] = importance_avg
        
        ba_val_list.append(ba_val)
        mcc_val_list.append(mcc_val)
        f1_val_list.append(f1_val)
        sen_val_list.append(sen_val)
        spe_val_list.append(spe_val)
        
        auc_dic_val[i] = {}
        auc_dic_val[i]['avg'] = {}
        auc_dic_val[i]['avg'] = auc_dic_val_avg
        
    
    # process the result
    if True:
        remove_key = []
        keys = list(importance_dic.keys())
        for i in keys:
            if type(i) != int:
                # convert to int
                importance_dic[int(i)] = importance_dic[i]
                # delete the original key
                remove_key.append(i)
                
        for i in remove_key:
            importance_dic.pop(i)

        # change all type of root of the value in auc_dic to list 
        for i in auc_dic:
            for j in auc_dic[i]:
                for k in auc_dic[i][j]:
                    auc_dic[i][j][k] = list(auc_dic[i][j][k]) 
            
        for i in auc_dic_val:
            for j in auc_dic_val[i]:
                for k in auc_dic_val[i][j]:
                    auc_dic_val[i][j][k] = list(auc_dic_val[i][j][k])
    
    # filter 
    if True:       
        average_auc_score_list = auc_plot(auc_dic, data = 'avg', cal = True)
        auc_score_list = [1 if i > threshold else 0 for i in average_auc_score_list]

        # only keep the trial that is larger than threshold for val 
        ba_val_list = [ba_val_list[i] for i in range(len(auc_score_list)) if auc_score_list[i] == 1]
        mcc_val_list = [mcc_val_list[i] for i in range(len(auc_score_list)) if auc_score_list[i] == 1] 
        f1_val_list = [f1_val_list[i] for i in range(len(auc_score_list)) if auc_score_list[i] == 1]
        sen_val_list = [sen_val_list[i] for i in range(len(auc_score_list)) if auc_score_list[i] == 1]
        spe_val_list = [spe_val_list[i] for i in range(len(auc_score_list)) if auc_score_list[i] == 1]

        auc_dic_val = {k: auc_dic_val[k] for k in range(len(auc_score_list)) if auc_score_list[k] == 1}

        # reorder the auc_dic_val
        new_key_list = list(auc_dic_val.keys())
        new_aic_dic_val = {}
        for i in range(len(new_key_list)):
            new_aic_dic_val[i] = auc_dic_val[new_key_list[i]]
            auc_dic_val.pop(new_key_list[i])

        # update the auc_dic_val
        auc_dic_val = new_aic_dic_val

        print(f"Number of trial that is larger than threshold: {len(auc_score_list)}")
    
    if save:
        # exp_log 
        exp_log[model_name]['ba_avg'] = list(ba_list)
        exp_log[model_name]['mcc_avg'] = list(mcc_list)
        exp_log[model_name]['f1_avg'] = list(f1_list)
        exp_log[model_name]['sen_avg'] = list(sen_list)
        exp_log[model_name]['spe_avg'] = list(spe_list)
        exp_log[model_name]['auc_dic'] = auc_dic
        exp_log[model_name]['importance_dic'] = importance_dic
        exp_log[model_name]['p_avg'] = list(p_avg_list)

        exp_log[model_name]['ba_val_avg'] = list(ba_val_list)
        exp_log[model_name]['mcc_val_avg'] = list(mcc_val_list)
        exp_log[model_name]['f1_val_avg'] = list(f1_val_list)
        exp_log[model_name]['sen_val_avg'] = list(sen_val_list)
        exp_log[model_name]['spe_val_avg'] = list(spe_val_list)
        exp_log[model_name]['auc_dic_val'] = auc_dic_val

        # save the log
        with open(f'Academic/A01_RIF/result/exp_log_{version}.json', 'w') as f:
            json.dump(exp_log, f, indent=4)
            
from sklearn.linear_model import Lasso

def train_evaluate_model_Lasso(data, target, random_seed, data_val = None):
        
        X_train, X_test, y_train, y_test = random_split_data(data, target, random_seed)
        
        # hyperparameter tuning for Lasso 
        hyperparameter = {'alpha': [0.0001,0.001, 0.05, 0.1, 0.5]} 
        from sklearn.model_selection import GridSearchCV
        grid = GridSearchCV(Lasso(), hyperparameter, refit=True, verbose=0)
        grid.fit(X_train, y_train)
        
        # best hyperparameter
        # print(grid.best_params_)
        
        model = Lasso(alpha=grid.best_params_['alpha'], random_state=random_seed)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
        probs = model.predict(X_test)
        
        threshold = 0.4
        y_pred = [1 if i > threshold else 0 for i in probs]
        # print(probs)
        
        ba, mcc, f1, cm, sen, spe = evaluation(y_test, y_pred)
        
        auc_dic = {}
        auc_dic['probs'] = probs
        auc_dic['y_test'] = y_test
        auc_dic['y_pred'] = y_pred
        auc_dic['X_test'] = [list(X_test.iloc[i]) for i in range(len(X_test))]
        
        # importance 
        importance = model.coef_
        
        if data_val is not None:
                X_val = data_val.drop(columns=[target])
                y_val = data_val[target]
                probs = model.predict(X_val)
                
                y_val_pred = [1 if i > threshold else 0 for i in probs]
                
                auc_dic_val = {}
                
                auc_dic_val['probs'] = probs
                auc_dic_val['y_test'] = y_val
                auc_dic_val['y_pred'] = y_val_pred
        
                ba_val, mcc_val, f1_val, cm_val, sen_val, spe_val = evaluation(y_val, y_val_pred)
                
                return {"test": [ba, mcc, f1, cm, sen, spe, auc_dic, grid.best_params_, importance], "val": [ba_val, mcc_val, f1_val, cm_val, sen_val, spe_val, auc_dic_val]}
        
        return ba, mcc, f1, cm, sen, spe, auc_dic,grid.best_params_, importance

# num of experiment 
num_exp = 50
model_name = 'Lasso'
threshold = 0
save = True

if True:
    ba_list = []
    mcc_list = []
    f1_list = []
    sen_list = []
    spe_list = []
    p_avg_list = []

    importance_dic = {}

    ba_val_list = []
    mcc_val_list = []
    f1_val_list = []
    sen_val_list = []
    spe_val_list = []

    auc_dic = {}
    auc_dic_val = {}
    

# check existence of result in exp_log
if 'auc_dic' not in exp_log[model_name].keys():
    # run the experiment
    for i in tqdm(range(num_exp)):
        random_seed = np.random.randint(0, 100)
        
        dic = train_evaluate_model_Lasso(data, target, random_seed, data_val)
        ba_avg, mcc_avg, f1_avg, cm_avg, sen_avg, spe_avg, auc_dic_avg, p_avg, importance_avg = dic['test']
        ba_val, mcc_val, f1_val, cm_val, sen_val, spe_val, auc_dic_val_avg = dic['val']
        
        ba_list.append(ba_avg)
        mcc_list.append(mcc_avg)
        f1_list.append(f1_avg)
        sen_list.append(sen_avg)
        spe_list.append(spe_avg)
        p_avg_list.append(p_avg)
        
        importance_avg = list(importance_avg.T @ X_feature_importance)
        importance_avg = [(col_name[i], importance_avg[i]) for i in range(len(importance_avg))]
        importance_avg = sorted(importance_avg, key=lambda x: abs(x[1]), reverse=True)
        
        auc_dic[i] = {}
        auc_dic[i]['avg'] = {}
        auc_dic[i]['avg'] = auc_dic_avg
        
        importance_dic[i] = {}
        importance_dic[i]['avg'] = {}
        importance_dic[i]['avg'] = importance_avg
        
        ba_val_list.append(ba_val)
        mcc_val_list.append(mcc_val)
        f1_val_list.append(f1_val)
        sen_val_list.append(sen_val)
        spe_val_list.append(spe_val)
        
        auc_dic_val[i] = {}
        auc_dic_val[i]['avg'] = {}
        auc_dic_val[i]['avg'] = auc_dic_val_avg
        
    
    # process the result
    if True:
        remove_key = []
        keys = list(importance_dic.keys())
        for i in keys:
            if type(i) != int:
                # convert to int
                importance_dic[int(i)] = importance_dic[i]
                # delete the original key
                remove_key.append(i)
                
        for i in remove_key:
            importance_dic.pop(i)

        # change all type of root of the value in auc_dic to list 
        for i in auc_dic:
            for j in auc_dic[i]:
                for k in auc_dic[i][j]:
                    auc_dic[i][j][k] = list(auc_dic[i][j][k]) 
            
        for i in auc_dic_val:
            for j in auc_dic_val[i]:
                for k in auc_dic_val[i][j]:
                    auc_dic_val[i][j][k] = list(auc_dic_val[i][j][k])
    
    # filter 
    if True:       
        average_auc_score_list = auc_plot(auc_dic, data = 'avg', cal = True)
        auc_score_list = [1 if i > threshold else 0 for i in average_auc_score_list]

        # only keep the trial that is larger than threshold for val 
        ba_val_list = [ba_val_list[i] for i in range(len(auc_score_list)) if auc_score_list[i] == 1]
        mcc_val_list = [mcc_val_list[i] for i in range(len(auc_score_list)) if auc_score_list[i] == 1] 
        f1_val_list = [f1_val_list[i] for i in range(len(auc_score_list)) if auc_score_list[i] == 1]
        sen_val_list = [sen_val_list[i] for i in range(len(auc_score_list)) if auc_score_list[i] == 1]
        spe_val_list = [spe_val_list[i] for i in range(len(auc_score_list)) if auc_score_list[i] == 1]

        auc_dic_val = {k: auc_dic_val[k] for k in range(len(auc_score_list)) if auc_score_list[k] == 1}

        # reorder the auc_dic_val
        new_key_list = list(auc_dic_val.keys())
        new_aic_dic_val = {}
        for i in range(len(new_key_list)):
            new_aic_dic_val[i] = auc_dic_val[new_key_list[i]]
            auc_dic_val.pop(new_key_list[i])

        # update the auc_dic_val
        auc_dic_val = new_aic_dic_val

        print(f"Number of trial that is larger than threshold: {len(auc_score_list)}")
    
    if save:
        # exp_log 
        exp_log[model_name]['ba_avg'] = list(ba_list)
        exp_log[model_name]['mcc_avg'] = list(mcc_list)
        exp_log[model_name]['f1_avg'] = list(f1_list)
        exp_log[model_name]['sen_avg'] = list(sen_list)
        exp_log[model_name]['spe_avg'] = list(spe_list)
        exp_log[model_name]['auc_dic'] = auc_dic
        exp_log[model_name]['importance_dic'] = importance_dic
        exp_log[model_name]['p_avg'] = list(p_avg_list)

        exp_log[model_name]['ba_val_avg'] = list(ba_val_list)
        exp_log[model_name]['mcc_val_avg'] = list(mcc_val_list)
        exp_log[model_name]['f1_val_avg'] = list(f1_val_list)
        exp_log[model_name]['sen_val_avg'] = list(sen_val_list)
        exp_log[model_name]['spe_val_avg'] = list(spe_val_list)
        exp_log[model_name]['auc_dic_val'] = auc_dic_val

        # save the log
        with open(f'Academic/A01_RIF/result/exp_log_{version}.json', 'w') as f:
            json.dump(exp_log, f, indent=4)
            
# Naive Bayes
from sklearn.naive_bayes import GaussianNB

def train_evaluate_model_NB(data, target, random_seed, data_val = None):
    
    X_train, X_test, y_train, y_test = random_split_data(data, target, random_seed)
    priors = np.array([len(y_train[y_train == 0])/len(y_train), len(y_train[y_train == 1])/len(y_train)])
    model = GaussianNB(priors = priors)
    model.fit(X_train, y_train)
    
    probs = model.predict_proba(X_test)[:,1]
    
    threshold = 0.5
    
    y_pred =  [1 if i > threshold else 0 for i in probs]
    
    ba, mcc, f1, cm, sen, spe = evaluation(y_test, y_pred)
    
    auc_dic = {}
    
    auc_dic['probs'] = probs
    auc_dic['y_test'] = y_test
    auc_dic['y_pred'] = y_pred
    auc_dic['X_test'] = [list(X_test.iloc[i]) for i in range(len(X_test))]
    
    importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=random_seed)['importances_mean']
    
    if data_val is not None:
        
        X_val = data_val.drop(columns=[target])
        y_val = data_val[target]
        probs = model.predict_proba(X_val)[:,1]
        
        y_val_pred = [1 if i > threshold else 0 for i in probs]
        
        auc_dic_val = {}
        
        auc_dic_val['probs'] = probs
        auc_dic_val['y_test'] = y_val
        auc_dic_val['y_pred'] = y_val_pred
        
        ba_val, mcc_val, f1_val, cm_val, sen_val, spe_val = evaluation(y_val, y_val_pred)
        
        return {"test": [ba, mcc, f1, cm, sen, spe, auc_dic, {}, importance], "val": [ba_val, mcc_val, f1_val, cm_val, sen_val, spe_val, auc_dic_val]}
    
    return ba, mcc, f1, cm, sen, spe, auc_dic, {}, importance

# num of experiment 
num_exp = 50
model_name = 'NaiveBayes'
threshold = 0
save = True

if True:
    ba_list = []
    mcc_list = []
    f1_list = []
    sen_list = []
    spe_list = []
    p_avg_list = []

    importance_dic = {}

    ba_val_list = []
    mcc_val_list = []
    f1_val_list = []
    sen_val_list = []
    spe_val_list = []

    auc_dic = {}
    auc_dic_val = {}
    

# check existence of result in exp_log
if 'auc_dic' not in exp_log[model_name].keys():
    # run the experiment
    for i in tqdm(range(num_exp)):
        random_seed = np.random.randint(0, 100)
        
        dic = train_evaluate_model_NB(data, target, random_seed, data_val)
        ba_avg, mcc_avg, f1_avg, cm_avg, sen_avg, spe_avg, auc_dic_avg, p_avg, importance_avg = dic['test']
        ba_val, mcc_val, f1_val, cm_val, sen_val, spe_val, auc_dic_val_avg = dic['val']
        
        ba_list.append(ba_avg)
        mcc_list.append(mcc_avg)
        f1_list.append(f1_avg)
        sen_list.append(sen_avg)
        spe_list.append(spe_avg)
        p_avg_list.append(p_avg)
        
        importance_avg = list(importance_avg.T @ X_feature_importance)
        importance_avg = [(col_name[i], importance_avg[i]) for i in range(len(importance_avg))]
        importance_avg = sorted(importance_avg, key=lambda x: abs(x[1]), reverse=True)
        
        auc_dic[i] = {}
        auc_dic[i]['avg'] = {}
        auc_dic[i]['avg'] = auc_dic_avg
        
        importance_dic[i] = {}
        importance_dic[i]['avg'] = {}
        importance_dic[i]['avg'] = importance_avg
        
        ba_val_list.append(ba_val)
        mcc_val_list.append(mcc_val)
        f1_val_list.append(f1_val)
        sen_val_list.append(sen_val)
        spe_val_list.append(spe_val)
        
        auc_dic_val[i] = {}
        auc_dic_val[i]['avg'] = {}
        auc_dic_val[i]['avg'] = auc_dic_val_avg
        
    
    # process the result
    if True:
        remove_key = []
        keys = list(importance_dic.keys())
        for i in keys:
            if type(i) != int:
                # convert to int
                importance_dic[int(i)] = importance_dic[i]
                # delete the original key
                remove_key.append(i)
                
        for i in remove_key:
            importance_dic.pop(i)

        # change all type of root of the value in auc_dic to list 
        for i in auc_dic:
            for j in auc_dic[i]:
                for k in auc_dic[i][j]:
                    auc_dic[i][j][k] = list(auc_dic[i][j][k]) 
            
        for i in auc_dic_val:
            for j in auc_dic_val[i]:
                for k in auc_dic_val[i][j]:
                    auc_dic_val[i][j][k] = list(auc_dic_val[i][j][k])
    
    # filter 
    if True:       
        average_auc_score_list = auc_plot(auc_dic, data = 'avg', cal = True)
        auc_score_list = [1 if i > threshold else 0 for i in average_auc_score_list]

        # only keep the trial that is larger than threshold for val 
        ba_val_list = [ba_val_list[i] for i in range(len(auc_score_list)) if auc_score_list[i] == 1]
        mcc_val_list = [mcc_val_list[i] for i in range(len(auc_score_list)) if auc_score_list[i] == 1] 
        f1_val_list = [f1_val_list[i] for i in range(len(auc_score_list)) if auc_score_list[i] == 1]
        sen_val_list = [sen_val_list[i] for i in range(len(auc_score_list)) if auc_score_list[i] == 1]
        spe_val_list = [spe_val_list[i] for i in range(len(auc_score_list)) if auc_score_list[i] == 1]

        auc_dic_val = {k: auc_dic_val[k] for k in range(len(auc_score_list)) if auc_score_list[k] == 1}

        # reorder the auc_dic_val
        new_key_list = list(auc_dic_val.keys())
        new_aic_dic_val = {}
        for i in range(len(new_key_list)):
            new_aic_dic_val[i] = auc_dic_val[new_key_list[i]]
            auc_dic_val.pop(new_key_list[i])

        # update the auc_dic_val
        auc_dic_val = new_aic_dic_val

        print(f"Number of trial that is larger than threshold: {len(auc_score_list)}")
    
    if save:
        # exp_log 
        exp_log[model_name]['ba_avg'] = list(ba_list)
        exp_log[model_name]['mcc_avg'] = list(mcc_list)
        exp_log[model_name]['f1_avg'] = list(f1_list)
        exp_log[model_name]['sen_avg'] = list(sen_list)
        exp_log[model_name]['spe_avg'] = list(spe_list)
        exp_log[model_name]['auc_dic'] = auc_dic
        exp_log[model_name]['importance_dic'] = importance_dic
        exp_log[model_name]['p_avg'] = list(p_avg_list)

        exp_log[model_name]['ba_val_avg'] = list(ba_val_list)
        exp_log[model_name]['mcc_val_avg'] = list(mcc_val_list)
        exp_log[model_name]['f1_val_avg'] = list(f1_val_list)
        exp_log[model_name]['sen_val_avg'] = list(sen_val_list)
        exp_log[model_name]['spe_val_avg'] = list(spe_val_list)
        exp_log[model_name]['auc_dic_val'] = auc_dic_val

        # save the log
        with open(f'Academic/A01_RIF/result/exp_log_{version}.json', 'w') as f:
            json.dump(exp_log, f, indent=4)
            
# neural network
from sklearn.neural_network import MLPClassifier

def train_evaluate_model_NN(data, target, random_seed, data_val = None):
    
    
    X_train, X_test, y_train, y_test = random_split_data(data, target, random_seed)
    
    # architecture of the neural network
    # two layers with 15 neurons each
    model = MLPClassifier(hidden_layer_sizes=(40,20,10), max_iter=1000, random_state=random_seed, early_stopping=True, n_iter_no_change=200, tol=1e-4, activation='relu', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001)
    
    model.fit(X_train, y_train)
    
    probs = model.predict_proba(X_test)[:,1]
    
    threshold = 0.5
    y_pred = [1 if i > threshold else 0 for i in probs]
    
    ba, mcc, f1, cm, sen, spe = evaluation(y_test, y_pred)
    
    auc_dic = {}
    
    auc_dic['probs'] = probs
    auc_dic['y_test'] = y_test
    auc_dic['y_pred'] = y_pred
    auc_dic['X_test'] = [list(X_test.iloc[i]) for i in range(len(X_test))]
    
    importance = model.coefs_
    
    if data_val is not None:
        X_val = data_val.drop(columns=[target])
        y_val = data_val[target]
        probs = model.predict_proba(X_val)[:,1]
        
        y_val_pred = [1 if i > threshold else 0 for i in probs]
        
        auc_dic_val = {}
        
        auc_dic_val['probs'] = probs
        auc_dic_val['y_test'] = y_val
        auc_dic_val['y_pred'] = y_val_pred
        
        ba_val, mcc_val, f1_val, cm_val, sen_val, spe_val = evaluation(y_val, y_val_pred)
        
        return {"test": [ba, mcc, f1, cm, sen, spe, auc_dic, {}, importance], "val": [ba_val, mcc_val, f1_val, cm_val, sen_val, spe_val, auc_dic_val]}
        
    return ba, mcc, f1, cm, sen, spe, auc_dic, {}, importance
    
# num of experiment 
num_exp = 50
model_name = 'NeuralNetwork'
threshold = 0
save = True

if True:
    ba_list = []
    mcc_list = []
    f1_list = []
    sen_list = []
    spe_list = []
    p_avg_list = []

    importance_dic = {}

    ba_val_list = []
    mcc_val_list = []
    f1_val_list = []
    sen_val_list = []
    spe_val_list = []

    auc_dic = {}
    auc_dic_val = {}
    

# check existence of result in exp_log
if 'auc_dic' not in exp_log[model_name].keys():
    # run the experiment
    for i in tqdm(range(num_exp)):
        random_seed = np.random.randint(0, 100)
        
        dic = train_evaluate_model_NN(data, target, random_seed, data_val)
        ba_avg, mcc_avg, f1_avg, cm_avg, sen_avg, spe_avg, auc_dic_avg, p_avg, importance_avg = dic['test']
        ba_val, mcc_val, f1_val, cm_val, sen_val, spe_val, auc_dic_val_avg = dic['val']
        
        ba_list.append(ba_avg)
        mcc_list.append(mcc_avg)
        f1_list.append(f1_avg)
        sen_list.append(sen_avg)
        spe_list.append(spe_avg)
        p_avg_list.append(p_avg)
        
        importance_avg = {}
        
        auc_dic[i] = {}
        auc_dic[i]['avg'] = {}
        auc_dic[i]['avg'] = auc_dic_avg
        
        importance_dic[i] = {}
        importance_dic[i]['avg'] = {}
        importance_dic[i]['avg'] = importance_avg
        
        ba_val_list.append(ba_val)
        mcc_val_list.append(mcc_val)
        f1_val_list.append(f1_val)
        sen_val_list.append(sen_val)
        spe_val_list.append(spe_val)
        
        auc_dic_val[i] = {}
        auc_dic_val[i]['avg'] = {}
        auc_dic_val[i]['avg'] = auc_dic_val_avg
        
    
    # process the result
    if True:
        remove_key = []
        keys = list(importance_dic.keys())
        for i in keys:
            if type(i) != int:
                # convert to int
                importance_dic[int(i)] = importance_dic[i]
                # delete the original key
                remove_key.append(i)
                
        for i in remove_key:
            importance_dic.pop(i)

        # change all type of root of the value in auc_dic to list 
        for i in auc_dic:
            for j in auc_dic[i]:
                for k in auc_dic[i][j]:
                    auc_dic[i][j][k] = list(auc_dic[i][j][k]) 
            
        for i in auc_dic_val:
            for j in auc_dic_val[i]:
                for k in auc_dic_val[i][j]:
                    auc_dic_val[i][j][k] = list(auc_dic_val[i][j][k])
    
    # filter 
    if True:       
        average_auc_score_list = auc_plot(auc_dic, data = 'avg', cal = True)
        auc_score_list = [1 if i > threshold else 0 for i in average_auc_score_list]

        # only keep the trial that is larger than threshold for val 
        ba_val_list = [ba_val_list[i] for i in range(len(auc_score_list)) if auc_score_list[i] == 1]
        mcc_val_list = [mcc_val_list[i] for i in range(len(auc_score_list)) if auc_score_list[i] == 1] 
        f1_val_list = [f1_val_list[i] for i in range(len(auc_score_list)) if auc_score_list[i] == 1]
        sen_val_list = [sen_val_list[i] for i in range(len(auc_score_list)) if auc_score_list[i] == 1]
        spe_val_list = [spe_val_list[i] for i in range(len(auc_score_list)) if auc_score_list[i] == 1]

        auc_dic_val = {k: auc_dic_val[k] for k in range(len(auc_score_list)) if auc_score_list[k] == 1}

        # reorder the auc_dic_val
        new_key_list = list(auc_dic_val.keys())
        new_aic_dic_val = {}
        for i in range(len(new_key_list)):
            new_aic_dic_val[i] = auc_dic_val[new_key_list[i]]
            auc_dic_val.pop(new_key_list[i])

        # update the auc_dic_val
        auc_dic_val = new_aic_dic_val

        print(f"Number of trial that is larger than threshold: {len(auc_score_list)}")
    
    if save:
        # exp_log 
        exp_log[model_name]['ba_avg'] = list(ba_list)
        exp_log[model_name]['mcc_avg'] = list(mcc_list)
        exp_log[model_name]['f1_avg'] = list(f1_list)
        exp_log[model_name]['sen_avg'] = list(sen_list)
        exp_log[model_name]['spe_avg'] = list(spe_list)
        exp_log[model_name]['auc_dic'] = auc_dic
        exp_log[model_name]['importance_dic'] = importance_dic
        exp_log[model_name]['p_avg'] = list(p_avg_list)

        exp_log[model_name]['ba_val_avg'] = list(ba_val_list)
        exp_log[model_name]['mcc_val_avg'] = list(mcc_val_list)
        exp_log[model_name]['f1_val_avg'] = list(f1_val_list)
        exp_log[model_name]['sen_val_avg'] = list(sen_val_list)
        exp_log[model_name]['spe_val_avg'] = list(spe_val_list)
        exp_log[model_name]['auc_dic_val'] = auc_dic_val

        # save the log
        with open(f'Academic/A01_RIF/result/exp_log_{version}.json', 'w') as f:
            json.dump(exp_log, f, indent=4)



######################################################################
# test
if __name__ == '__main__':
    # example 
    # os.system('python Academic/A01_RIF/method/result_generation.py --exp_version v1')
    pass