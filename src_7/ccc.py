# import os
# import math
# import numpy as np
# import random
# from scipy import sparse
# import scipy.io as scio
# import argparse 
# import optuna
# from sklearn.preprocessing import normalize
# from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
# from sklearn.metrics.pairwise import cosine_similarity
# import torch
# from torch_geometric.utils import from_scipy_sparse_matrix
# from torch_geometric.data import Data
# from pygod.detector import AdONE
# from utils.outlier_inject_cf import *

# def evaluate_outlier_detection(y_true, y_pred, sens):
#     if isinstance(y_true, torch.Tensor):
#         y_true = y_true.numpy()
#     if isinstance(y_pred, torch.Tensor):
#         y_pred = y_pred.numpy()
#     if isinstance(sens, torch.Tensor):
#         sens = sens.numpy()

#     acc = accuracy_score(y_true, y_pred)
#     auc_roc = roc_auc_score(y_true, y_pred)
#     f1_s = f1_score(y_true, y_pred)
#     dp = np.abs(np.mean(y_pred[sens == 0]) - np.mean(y_pred[sens == 1]))
#     eoo = np.abs(np.mean(y_pred[(sens == 0) & (y_true == 1)]) - np.mean(y_pred[(sens == 1) & (y_true == 1)]))
#     eval_results = {'dp': dp, 'eoo': eoo, 'auc': auc_roc, 'f1': f1_s, 'acc': acc}
#     return eval_results

# def load_synthetic(path, label_number=1000):
#     data = scio.loadmat(path)
#     features = data['x']
#     features_cf = data['x_cf']
#     adj = data['adj']
#     adj_cf = data['adj_cf']
#     labels = data['y']
#     labels_cf = data['y_cf']
#     sens = data['sens'][0]
#     sens_cf = data['sens_cf'][0]
#     features = np.concatenate([sens.reshape(-1,1), features], axis=1)
#     raw_data_info = {}
#     raw_data_info['adj'] = adj
#     raw_data_info['w'] = data['w']
#     raw_data_info['w_s'] = data['w_s'][0][0]
#     raw_data_info['z'] = data['z']
#     raw_data_info['v'] = data['v']
#     raw_data_info['feat_idxs'] = data['feat_idxs'][0]
#     raw_data_info['alpha'] = data['alpha'][0][0]
#     features = torch.FloatTensor(features)
#     features_cf = torch.FloatTensor(features_cf)

#     return sens, adj, features, labels, sens_cf, adj_cf, features_cf, labels_cf, raw_data_info

# def generate_synthetic_data(path, flip_rate):
#     n = 2000
#     z_dim = 50
#     dim = 25
#     p = 0.4
#     alpha = 0.01  
#     threshold = 0.4
    
#     sens = np.random.binomial(n=1, p=p, size=n)
#     embedding = np.random.normal(loc=0, scale=1, size=(n, z_dim))
#     feat_idxs = random.sample(range(z_dim), dim)
#     v = np.random.normal(0, 1, size=(1, dim))
#     features = embedding[:, feat_idxs] + (np.dot(sens.reshape(-1,1), v))  # (n x dim) + (1 x dim) -> n x dim

#     adj = np.zeros((n, n))
#     sens_sim = np.zeros((n, n))
#     for i in range(n):
#         for j in range(i, n):  # i<=j
#             if i == j:
#                 sens_sim[i][j] = 1
#                 continue
#             sens_sim[i][j] = sens_sim[j][i] = (sens[i] == sens[j])

#     similarities = cosine_similarity(embedding)  # n x n
#     adj = similarities + alpha * sens_sim

#     print('adj max: ', adj.max(), ' min: ', adj.min())
#     adj[np.where(adj >= threshold)] = 1
#     adj[np.where(adj < threshold)] = 0
#     adj = sparse.csr_matrix(adj)

#     w = np.random.normal(0, 1, size=(z_dim, 1))
#     w_s = 1    
#     adj_norm = normalize(adj, norm='l1', axis=1)
#     nb_sens_ave = adj_norm @ sens  # n x 1, average sens of 1-hop neighbors

#     dd = np.matmul(embedding, w)
#     d2 = nb_sens_ave.reshape(-1,1)
#     labels = dd + w_s * d2 # n x 1
#     labels = labels.reshape(-1)
#     labels_mean = np.mean(labels)
#     labels_binary = np.zeros_like(labels)
#     labels_binary[np.where(labels > labels_mean)] = 1.0
    

#     sampled_idx = random.sample(range(n), int(flip_rate * n))
#     # data_cf = data.clone()
#     sens_cf = sens.copy()
#     sens_cf[sampled_idx] = 1 - sens[sampled_idx]
#     embedding_cf = embedding
#     feat_idxs_cf = feat_idxs
#     features_cf = embedding_cf[:, feat_idxs_cf] + (np.dot(sens_cf.reshape(-1,1), v))
#     adj_cf = np.zeros((n, n))
#     sens_sim_cf = np.zeros((n, n))
#     for i in range(n):
#         for j in range(i, n):  # i<=j
#             if i == j:
#                 sens_sim_cf[i][j] = 1
#                 continue
#             sens_sim_cf[i][j] = sens_sim_cf[j][i] = (sens_cf[i] == sens_cf[j])
            
#     similarities_cf = similarities
#     adj_cf = similarities_cf + alpha * sens_sim_cf
    
#     adj_cf[np.where(adj_cf >= threshold)] = 1
#     adj_cf[np.where(adj_cf < threshold)] = 0
#     adj_cf = sparse.csr_matrix(adj_cf)
    
#     adj_norm_cf = normalize(adj_cf, norm='l1', axis=1)
#     nb_sens_ave_cf = adj_norm_cf @ sens_cf  # n x 1, average sens of 1-hop neighbors

#     dd_cf = np.matmul(embedding_cf, w)
#     d2_cf = nb_sens_ave_cf.reshape(-1,1)
#     labels_cf = dd_cf + w_s * d2_cf # n x 1
#     labels_cf = labels_cf.reshape(-1)
#     labels_mean_cf = np.mean(labels_cf)
#     labels_binary_cf = np.zeros_like(labels_cf)
#     labels_binary_cf[np.where(labels_cf > labels_mean_cf)] = 1.0
    
#     data = {'x': features, 'adj': adj, 'sens': sens, 'x_cf': features_cf, 'adj_cf': adj_cf, 'sens_cf': sens_cf, 'z': embedding, 'v': v, 'feat_idxs': feat_idxs, 'alpha': alpha, 'w': w, 'w_s': w_s, 'y': labels_binary, 'y_cf': labels_binary_cf}
#     scio.savemat(path, data)
#     print('data saved in ', path)
#     return data


# def objective(trial, data, data_cf, sens, sens_cf):
#     lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
#     dropout = trial.suggest_float("dropout", 1e-6, 1, log=True)
#     weight_decay = trial.suggest_float("weight_decay", 1e-6, 1, log=True)
#     num_layers = trial.suggest_categorical("num_layers", [2, 3, 4, 5])
#     hid_dim = trial.suggest_categorical("hid_dim", [32, 64, 128, 256])
#     def train_and_predict(data, lr, dropout, weight_decay, num_layers, hid_dim, contamination=0.05, threshold_percentile=95):
#         model = AdONE(lr=lr, dropout=dropout, weight_decay=weight_decay, num_layers=num_layers, hid_dim=hid_dim, contamination=contamination)
#         model.fit(data)
#         y_score = np.array(model.decision_score_).ravel()
#         threshold = np.percentile(y_score, threshold_percentile)
#         y_pred = (y_score > threshold).astype(int)
#         return y_score, y_pred, threshold
#     y_score, y_pred, threshold = train_and_predict(data, lr, dropout, weight_decay, num_layers, hid_dim)
#     y_score_cf, y_pred_cf, threshold_cf = train_and_predict(data_cf, lr, dropout, weight_decay, num_layers, hid_dim)
#     eval = evaluate_outlier_detection(data.y, y_pred, sens)
#     eval_cf = evaluate_outlier_detection(data.y, y_pred_cf, sens_cf)
    
#     cf = 1 - (np.sum(y_pred_cf == y_pred) / 2000)
#     eval['cf'] = cf
#     print(f"Trial {trial.number}   :", eval)
#     print(f"Trial {trial.number} cf:", eval_cf)
#     auc = (eval['auc'] + eval_cf['auc'])/2
#     return auc


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--outlier_type', type=str, choices=['structural', 'contextual', 'dice', 'path', 'cont_struc', 'path_dice'] , required=True, help='Model to use')
#     parser.add_argument('--flip_rate', type=float)
#     args = parser.parse_args()
    
#     sens_idx = 0
#     label_number = 1000 
#     # generate_synthetic_data(f'synthetic/synthetic_{args.flip_rate}.mat', args.flip_rate)
#     path_sythetic = f'synthetic/synthetic_{args.flip_rate}.mat'
#     sens, adj, features, labels, sens_cf, adj_cf, features_cf, labels_cf, raw_data_info = load_synthetic(path=path_sythetic, label_number=label_number)
#     edge_index, edge_attr = from_scipy_sparse_matrix(adj)
#     edge_index_cf, edge_attr_cf = from_scipy_sparse_matrix(adj_cf)
#     data = Data(x=features, edge_index=edge_index, y=labels)
#     data_cf = Data(x=features_cf, edge_index=edge_index_cf, y=labels_cf)
    


#     args.struc_drop_prob = 0.2
#     args.dice_ratio = 0.5
#     args.outlier_seed = 0
#     args.struc_clique_size = math.ceil(np.mean(np.sum(adj, axis=1)))
#     args.sample_size = args.struc_clique_size
#     args.outlier_num  = math.ceil(data.num_nodes * 0.05)
    
#     data_mod, data_mod_cf, data_mod.y = outlier_injection_cf(args, data, data_cf)


#     study = optuna.create_study(direction="maximize")
#     study.optimize(lambda trial: objective(trial, data_mod, data_mod_cf, sens, sens_cf), n_trials=30)

#     print("Best trial:")
#     trial = study.best_trial

#     print("  Value: ", trial.value)
#     print("  Params: ")
#     for key, value in trial.params.items():
#         print("    {}: {}".format(key, value))
        
    
    

import os
import math
import numpy as np
import random
from scipy import sparse
import scipy.io as scio
import argparse 
import optuna
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
import torch
from torch_geometric.utils import from_scipy_sparse_matrix
from torch_geometric.data import Data
from pygod.detector import AdONE
from utils.outlier_inject_cf import *

def evaluate_outlier_detection(y_true, y_pred, sens):
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.numpy()
    if isinstance(sens, torch.Tensor):
        sens = sens.numpy()

    acc = accuracy_score(y_true, y_pred)
    auc_roc = roc_auc_score(y_true, y_pred)
    f1_s = f1_score(y_true, y_pred)
    dp = np.abs(np.mean(y_pred[sens == 0]) - np.mean(y_pred[sens == 1]))
    eoo = np.abs(np.mean(y_pred[(sens == 0) & (y_true == 1)]) - np.mean(y_pred[(sens == 1) & (y_true == 1)]))
    eval_results = {'dp': dp, 'eoo': eoo, 'auc': auc_roc, 'f1': f1_s, 'acc': acc}
    return eval_results

def load_synthetic(path, label_number=1000):
    data = scio.loadmat(path)
    features = data['x']
    features_cf = data['x_cf']
    adj = data['adj']
    adj_cf = data['adj_cf']
    labels = data['y']
    labels_cf = data['y_cf']
    sens = data['sens'][0]
    sens_cf = data['sens_cf'][0]
    features = np.concatenate([sens.reshape(-1,1), features], axis=1)
    features_cf = np.concatenate([sens_cf.reshape(-1,1), features_cf], axis=1)
    raw_data_info = {}
    raw_data_info['adj'] = adj
    raw_data_info['w'] = data['w']
    raw_data_info['w_s'] = data['w_s'][0][0]
    raw_data_info['z'] = data['z']
    raw_data_info['v'] = data['v']
    raw_data_info['feat_idxs'] = data['feat_idxs'][0]
    raw_data_info['alpha'] = data['alpha'][0][0]
    features = torch.FloatTensor(features)
    features_cf = torch.FloatTensor(features_cf)

    return sens, adj, features, labels, sens_cf, adj_cf, features_cf, labels_cf, raw_data_info

def generate_synthetic_data(path, flip_rate):
    n = 2000
    z_dim = 50
    dim = 25
    p = 0.4
    alpha = 0.01  
    threshold = 0.4
    
    sens = np.random.binomial(n=1, p=p, size=n)
    embedding = np.random.normal(loc=0, scale=1, size=(n, z_dim))
    feat_idxs = random.sample(range(z_dim), dim)
    v = np.random.normal(0, 1, size=(1, dim))
    features = embedding[:, feat_idxs] + (np.dot(sens.reshape(-1,1), v))  # (n x dim) + (1 x dim) -> n x dim

    adj = np.zeros((n, n))
    sens_sim = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):  # i<=j
            if i == j:
                sens_sim[i][j] = 1
                continue
            sens_sim[i][j] = sens_sim[j][i] = (sens[i] == sens[j])

    similarities = cosine_similarity(embedding)  # n x n
    adj = similarities + alpha * sens_sim

    print('adj max: ', adj.max(), ' min: ', adj.min())
    adj[np.where(adj >= threshold)] = 1
    adj[np.where(adj < threshold)] = 0
    adj = sparse.csr_matrix(adj)

    w = np.random.normal(0, 1, size=(z_dim, 1))
    w_s = 1    
    adj_norm = normalize(adj, norm='l1', axis=1)
    nb_sens_ave = adj_norm @ sens  # n x 1, average sens of 1-hop neighbors

    dd = np.matmul(embedding, w)
    d2 = nb_sens_ave.reshape(-1,1)
    labels = dd + w_s * d2 # n x 1
    labels = labels.reshape(-1)
    labels_mean = np.mean(labels)
    labels_binary = np.zeros_like(labels)
    labels_binary[np.where(labels > labels_mean)] = 1.0
    

    sampled_idx = random.sample(range(n), int(flip_rate * n))
    # data_cf = data.clone()
    sens_cf = sens.copy()
    sens_cf[sampled_idx] = 1 - sens[sampled_idx]
    embedding_cf = embedding.copy()
    feat_idxs_cf = feat_idxs
    features_cf = embedding_cf[:, feat_idxs_cf] + (np.dot(sens_cf.reshape(-1,1), v))
    adj_cf = np.zeros((n, n))
    sens_sim_cf = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):  # i<=j
            if i == j:
                sens_sim_cf[i][j] = 1
                continue
            sens_sim_cf[i][j] = sens_sim_cf[j][i] = (sens_cf[i] == sens_cf[j])
            
    similarities_cf = similarities
    adj_cf = similarities_cf + alpha * sens_sim_cf
    
    adj_cf[np.where(adj_cf >= threshold)] = 1
    adj_cf[np.where(adj_cf < threshold)] = 0
    adj_cf = sparse.csr_matrix(adj_cf)
    
    adj_norm_cf = normalize(adj_cf, norm='l1', axis=1)
    nb_sens_ave_cf = adj_norm_cf @ sens_cf  # n x 1, average sens of 1-hop neighbors

    dd_cf = np.matmul(embedding_cf, w)
    d2_cf = nb_sens_ave_cf.reshape(-1,1)
    labels_cf = dd_cf + w_s * d2_cf # n x 1
    labels_cf = labels_cf.reshape(-1)
    labels_mean_cf = np.mean(labels_cf)
    labels_binary_cf = np.zeros_like(labels_cf)
    labels_binary_cf[np.where(labels_cf > labels_mean_cf)] = 1.0
    
    data = {'x': features, 'adj': adj, 'sens': sens, 'x_cf': features_cf, 'adj_cf': adj_cf, 'sens_cf': sens_cf, 'z': embedding, 'v': v, 'feat_idxs': feat_idxs, 'alpha': alpha, 'w': w, 'w_s': w_s, 'y': labels_binary, 'y_cf': labels_binary_cf}
    scio.savemat(path, data)
    print('data saved in ', path)
    return data


def objective(trial, data_list, data_cf_list, sens, sens_cf):
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    dropout = trial.suggest_float("dropout", 1e-6, 1, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1, log=True)
    num_layers = trial.suggest_categorical("num_layers", [2, 3, 4, 5])
    hid_dim = trial.suggest_categorical("hid_dim", [32, 64, 128, 256])
    
    def train_and_predict(data, lr, dropout, weight_decay, num_layers, hid_dim, contamination=0.05, threshold_percentile=95):
        model = AdONE(lr=lr, dropout=dropout, weight_decay=weight_decay, num_layers=num_layers, hid_dim=hid_dim, contamination=contamination)
        model.fit(data)
        y_score = np.array(model.decision_score_).ravel()
        threshold = np.percentile(y_score, threshold_percentile)
        y_pred = (y_score > threshold).astype(int)
        return y_score, y_pred, threshold
    
    outlier_types = ['structural', 'contextual', 'dice', 'path', 'cont_struc', 'path_dice']
    eval_result, eval_result_cf, auc = {}, {}, 0
    for outlier_type in outlier_types:
        data = data_list[outlier_type]
        data_cf = data_cf_list[outlier_type]
        y_score, y_pred, threshold = train_and_predict(data, lr, dropout, weight_decay, num_layers, hid_dim)
        y_score_cf, y_pred_cf, threshold_cf = train_and_predict(data_cf, lr, dropout, weight_decay, num_layers, hid_dim)
        eval_result[outlier_type] = evaluate_outlier_detection(data.y, y_pred, sens)
        eval_result_cf[outlier_type] = evaluate_outlier_detection(data.y, y_pred_cf, sens_cf)
        cf = 1 - (np.sum(y_pred_cf == y_pred) / 2000)
        eval_result[outlier_type]['cf'] = cf
        print(f"Trial {trial.number} {outlier_type}   :", eval_result[outlier_type])
        print(f"Trial {trial.number} {outlier_type} cf:", eval_result_cf[outlier_type])
        auc += (eval_result[outlier_type]['auc'] + eval_result_cf[outlier_type]['auc'])/12
    return auc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--outlier_type', type=str, choices=['structural', 'contextual', 'dice', 'path', 'cont_struc', 'path_dice'] , required=True, help='Model to use')
    parser.add_argument('--flip_rate', type=float)
    args = parser.parse_args()
    
    sens_idx = 0
    label_number = 1000 
    # generate_synthetic_data(f'synthetic/synthetic_{args.flip_rate}.mat', args.flip_rate)
    path_sythetic = f'synthetic/synthetic_{args.flip_rate}.mat'
    sens, adj, features, labels, sens_cf, adj_cf, features_cf, labels_cf, raw_data_info = load_synthetic(path=path_sythetic, label_number=label_number)
    edge_index, edge_attr = from_scipy_sparse_matrix(adj)
    edge_index_cf, edge_attr_cf = from_scipy_sparse_matrix(adj_cf)
    data = Data(x=features, edge_index=edge_index, y=labels)
    data_cf = Data(x=features_cf, edge_index=edge_index_cf, y=labels_cf)


    args.struc_drop_prob = 0.2
    args.dice_ratio = 0.5
    args.outlier_seed = 0
    args.struc_clique_size = math.ceil(np.mean(np.sum(adj, axis=1)))
    args.sample_size = args.struc_clique_size
    args.outlier_num  = math.ceil(data.num_nodes * 0.05)
    
    outlier_types = ['structural', 'contextual', 'dice', 'path', 'cont_struc', 'path_dice']
    data_mod_list, data_mod_cf_list = {}, {}
    for outlier_type in outlier_types:
        args.outlier_type = outlier_type
        data_mod_list[outlier_type], data_mod_cf_list[outlier_type], data_mod_list[outlier_type].y = outlier_injection_cf(args, data, data_cf)


    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, data_mod_list, data_mod_cf_list, sens, sens_cf), n_trials=50)

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
        
    
    
