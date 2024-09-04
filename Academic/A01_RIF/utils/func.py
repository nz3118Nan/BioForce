# system setting
import os
import sys
file_path = os.path.abspath(__file__)
root_dir = os.path.abspath(os.path.join(file_path, '..', '..', '..', '..'))
sys.path.append(root_dir)
sys.dont_write_bytecode = True
os.chdir(root_dir)

import pandas as pd
from tqdm import tqdm   
import numpy as np
import seaborn as sns
# roc curve
from sklearn.metrics import roc_curve, roc_auc_score

# import plt 
import matplotlib.pyplot as plt
##############################################################################################################

# KL divergence, Bhattacharyya distance, Information-divergence, KS-test
# to calculate Measure for Separability of Two Classes (MS)
# unordered two data sets
# .mutual_info_score
from sklearn.decomposition import PCA

# 没啥用 用于计算两个数据集的分离度
def MS(data, target, exp_num = 5000):
    data_tmp = data.copy()
    # X 
    X = data_tmp.drop(target, axis=1)
    y = data_tmp[target]
    # pca 99%
    pca = PCA(n_components=0.90, svd_solver='full')
    X = pca.fit_transform(X)
    
    # combine X with y
    data_tmp = pd.concat([pd.DataFrame(X), y], axis=1)
    
    X = data_tmp.drop(target, axis=1)
    
    # add a value to avoid log(negative value)
    # add a small value to avoid log(0)
    X = X - (min(X.min()) if min(X.min()) < 0 else 0) + 1e-10
    
    # update data_tmp
    data_tmp = pd.concat([X, y], axis=1)
    
    X_0 = data_tmp[data_tmp[target] == 0].drop(target, axis=1)
    X_1 = data_tmp[data_tmp[target] == 1].drop(target, axis=1)
    
    X_0 = np.array(X_0)
    X_1 = np.array(X_1)
    
    KL_list = []
    BC_list = []
    ID_list = []
    KS_list = []
    
    for i in range(exp_num):
        # sample
        X_0_sample = X_0[np.random.choice(len(X_0), 1)][0]
        X_1_sample = X_1[np.random.choice(len(X_1), 1)][0]
        
        # KL divergence for multivariate distribution
        KL = 0
        for j in range(X_0_sample.shape[0]):
            KL += np.sum(X_0_sample[j] * np.log(X_0_sample[j] / X_1_sample[j]))
            
        KL = KL / np.sqrt(X_0_sample.shape[0])
        
        # Bhattacharyya distance
        BC = 0
        for j in range(X_0_sample.shape[0]):
            BC += np.sqrt(X_0_sample[j] * X_1_sample[j])
        BC = BC/np.sqrt(X_0_sample.shape[0])
        BC = -np.log(BC)
        
        # Information-divergence
        ID = 0
        for j in range(X_0_sample.shape[0]):
            ID += X_0_sample[j] * np.log(2 * X_0_sample[j] / (X_0_sample[j] + X_1_sample[j]))
        ID = ID / np.sqrt(X_0_sample.shape[0])
        
        # KS-test
        KS = 0
        for j in range(X_0_sample.shape[0]):
            KS += np.abs(X_0_sample[j] - X_1_sample[j])
        KS = KS / np.sqrt(X_0_sample.shape[0])
        
        
    
        KL_list.append(KL)
        BC_list.append(BC)
        ID_list.append(ID)
        KS_list.append(KS)
        
    return np.mean(KL_list), np.mean(BC_list), np.mean(ID_list), np.mean(KS_list)


# plot the roc curve， calculate the average roc curve， calculate the argmax of mcc
def auc_plot(auc_dic, data = '0', cal = False, metric = False, index_independent = "mcc", weight = 1):
    if cal == False and metric == False:
        plt.figure(figsize=(10, 3))

    tpr_avg = [0] * 20
    fpr_avg = [0] * 20

    auc_score_list = []
    num_exp = len(auc_dic)
    
    balanance_accuracy_list_final = []
    mcc_list_final = []
    f1_list_final = []
    sen_list_final = []
    spe_list_final = []
    
    for i in range(num_exp):
        
        fpr, tpr, _ = roc_curve(auc_dic[i][data]['y_test'], auc_dic[i][data]['probs'])


        # stretch the line to 20 points add to the list, using ratio to extend the line 
        fpr = list(fpr)
        tpr = list(tpr)
        
        # number of fpr
        num_tmp = len(fpr)
        
        balanance_accuracy_list = []
        mcc_list = []
        f1_list = []
        sen_list = []
        spe_list = []
        
        for j in range(num_tmp):
            # balance accuracy
            ba = (1 + tpr[j] - fpr[j]) / 2
            balanance_accuracy_list.append(ba)
            
            
            # mcc
            if tpr[j] == 0:
                if fpr[j] == 0:
                    mcc = 0
                elif fpr[j] == 1:
                    mcc = -1
            elif tpr[j] == 1:
                if fpr[j] == 0:
                    mcc = 1
                    
            else:
                mcc = (tpr[j] - fpr[j]) / np.sqrt((tpr[j] + fpr[j]) * (1 - tpr[j] + 1 - fpr[j]))
                
            if tpr[j] == 0:
                tpr[j] = 1e-10
            if fpr[j] == 0:
                fpr[j] = 1e-10
            if tpr[j] == 1:
                tpr[j] = 1 - 1e-10
            if fpr[j] == 1:
                fpr[j] = 1 - 1e-10
        
            mcc_list.append(mcc)
            
            # f1
            f1 = 2 * tpr[j] / (2 * tpr[j] + fpr[j] + (1 - tpr[j]))
            
            f1_list.append(f1)
            
            # sensitivity
            sen_list.append(tpr[j])
            
            # specificity
            spe_list.append(1 - fpr[j])
            
        # select the best according to the mcc
        if index_independent == "mcc":
            index = np.argmax(mcc_list)
        elif index_independent == "spe":
            tmp_list = [spe_list[e] * weight + mcc_list[e] for e in range(len(mcc_list))]
            index = np.argmax(tmp_list)
        
        balanance_accuracy_list_final.append(balanance_accuracy_list[index])
        mcc_list_final.append(mcc_list[index])
        f1_list_final.append(f1_list[index])
        sen_list_final.append(sen_list[index])
        spe_list_final.append(spe_list[index])
        
        ## proprotionally extend the line to 20 points
        #using the ratio to extend the line to 20 points
        fpr = [fpr[int(i / 20 * len(fpr))] for i in range(20)]
        tpr = [tpr[int(i / 20 * len(tpr))] for i in range(20)]

        # add to the average line
        tpr_avg = [x + y for x, y in zip(tpr_avg, tpr)]
        fpr_avg = [x + y for x, y in zip(fpr_avg, fpr)]
        
        auc_score_list.append(roc_auc_score(auc_dic[i][data]['y_test'], auc_dic[i][data]['probs']))
        
        if cal == True:
            continue
        
        if metric == True:
            continue
        plt.plot(fpr, tpr, color='gray', alpha=0.5)
        
    if cal == True:
        return auc_score_list
    
    # print(balanance_accuracy_list_final, mcc_list_final, f1_list_final, sen_list_final, spe_list_final)
    if metric == True:
        return balanance_accuracy_list_final, mcc_list_final, f1_list_final, sen_list_final, spe_list_final
        
    # average line
    tpr_avg = [x / num_exp for x in tpr_avg]
    fpr_avg = [x / num_exp for x in fpr_avg]
    
    # calculate auc for average line using tpr_avg_1 and fpr_avg_1
    auc_avg = np.mean(auc_score_list)
    
    # add legend with auc score
    plt.plot(fpr_avg, tpr_avg, label=f'Model {data}, AUC: {round(auc_avg, 2)}')
    
    plt.title(f'ROC Curve for Model {data}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()


# evaluation 
# balance accuracy, mcc, f1, confusion matrix, sensitivity, specificity, auc
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef, f1_score, confusion_matrix




def evaluation(y_true, y_pred):
    ba = balanced_accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    # sensitivity
    sen = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    
    # specificity
    spe = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    
    return ba, mcc, f1, cm, sen, spe
