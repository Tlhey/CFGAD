import os
import copy
import numpy as np
import torch
import math
import argparse
from utils.loader_HNN_GAD import *
from utils.outlier_inject_cf import *
from torch_geometric.utils import dense_to_sparse

import optuna
from pygod.detector import AdONE
import torch.optim as optim

import ray
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix
# import tuned_config
from utils.Preprocessing import *
from utils.CFGT import Subgraph
from collections import defaultdict

def generate_synthetic_data(path, n, z_dim, p, q, alpha, beta, threshold, dim):    
    n = 2000
    z_dim = 50
    p = 0.4
    q = 0.3
    alpha = 0.01  
    beta = 0.01
    threshold = 0.6
    dim = 32
    sens = np.random.binomial(n=1, p=p, size=n)
    sens_repeat = np.repeat(sens.reshape(-1, 1), z_dim, axis=1)
    sens_embedding = np.random.normal(loc=sens_repeat, scale=1, size=(n, z_dim))
    labels = np.random.binomial(n=1, p=q, size=n)
    labels_repeat = np.repeat(labels.reshape(-1, 1), z_dim, axis=1)
    labels_embedding = np.random.normal(loc=labels_repeat, scale=1, size=(n, z_dim))
    features_embedding = np.concatenate((sens_embedding, labels_embedding), axis=1)
    weight = np.random.normal(loc=0, scale=1, size=(z_dim*2, dim))
    # features = np.matmul(features_embedding, weight)
    features = np.matmul(features_embedding, weight) + np.random.normal(loc=0, scale=1, size=(n, dim))

    adj = np.zeros((n, n))
    sens_sim = np.zeros((n, n))
    labels_sim = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):  # i<=j
            if i == j:
                sens_sim[i][j] = -1
                labels_sim[i][j] = -1
                continue
            sens_sim[i][j] = sens_sim[j][i] = (sens[i] == sens[j])
            labels_sim[i][j] = labels_sim[j][i] = (labels[i] == labels[j])
            # sim_ij = 1 - spatial.distance.cosine(embedding[i], embedding[j])  # [-1, 1]
            # adj[i][j] = adj[j][i] = sim_ij + alpha * (sens[i] == sens[j])

    similarities = cosine_similarity(features_embedding)  # n x n
    similarities[np.arange(n), np.arange(n)] = -1
    adj = similarities + alpha * sens_sim + beta * labels_sim
    print('adj max: ', adj.max(), ' min: ', adj.min())
    adj[np.where(adj >= threshold)] = 1
    adj[np.where(adj < threshold)] = 0
    edge_index, edge_attr = dense_to_sparse(torch.tensor(adj, dtype=torch.float))
    edge_num = adj.sum()
    # adj = sparse.csr_matrix(adj)
    # features = np.concatenate((sens.reshape(-1,1), features), axis=1)

    # generate counterfactual
    sens_flip = 1 - sens
    sens_flip_repeat = np.repeat(sens_flip.reshape(-1, 1), z_dim, axis=1)
    # sens_flip_embedding = np.random.normal(loc=sens_flip_repeat, scale=1, size=(n, z_dim))
    sens_flip_embedding = sens_embedding
    features_embedding = np.concatenate((sens_flip_embedding, labels_embedding), axis=1)
    features_cf = np.matmul(features_embedding, weight) + np.random.normal(loc=0, scale=1, size=(n, dim))

    adj_cf = np.zeros((n, n))
    sens_cf_sim = np.zeros((n, n))
    labels_cf_sim = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            if i == j:
                sens_cf_sim[i][j] = -1
                labels_cf_sim[i][j] = -1
                continue
            sens_cf_sim[i][j] = sens_cf_sim[j][i] = (sens_flip[i] == sens_flip[j])
            labels_cf_sim[i][j] = labels_cf_sim[j][i] = (labels[i] == labels[j])
    
    similarities_cf = cosine_similarity(features_cf)  # n x n
    similarities_cf[np.arange(n), np.arange(n)] = -1
    adj_cf = similarities_cf + alpha * sens_cf_sim + beta * labels_cf_sim
    print('adj_cf max', adj_cf.max(), ' min: ', adj_cf.min())
    adj_cf[np.where(adj_cf >= threshold)] = 1
    adj_cf[np.where(adj_cf < threshold)] = 0
    edge_index_cf, edge_attr_cf = dense_to_sparse(torch.tensor(adj_cf, dtype=torch.float))
    # edge_index_cf = torch.nonzero(torch.from_numpy(adj_cf)).t().contiguous()
    # adj_cf = sparse.csr_matrix(adj_cf)
    # features_cf = np.concatenate((sens_flip.reshape(-1,1), features_cf), axis=1)

    # statistics
    # pre_analysis(adj, labels, sens)
    # print('edge num: ', edge_num)
    data = {'x': features, 'edge_index': edge_index, 'labels': labels, 'sens': sens, 'x_cf': features_cf, 'edge_index_cf': edge_index_cf, "edge_num": edge_num}
    scio.savemat(path, data)
    # print("(labels_cf_sim - labels_sim", (labels_cf_sim - labels_sim.sum()))
    # print('data saved in ', path)
    return data

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


def objective(trial, data, sens):
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-1)
    num_layers = trial.suggest_int("num_layers", 1, 4)
    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128, 256])
    
    model = AdONE(lr=lr, num_layers=num_layers, hidden_dim=hidden_dim)
    model.fit(data)
    y_pred = model.decision_score_
    eval_results = evaluate_outlier_detection(data.y, y_pred, sens)
    auc_roc = eval_results['auc']
    return auc_roc

    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', type=str, choices=['credit', 'german', 'bail', 'pokec_z', 'pokec_n', 'UCSD34', 'synthetic'], required=True, help='Dataset to use')
    # parser.add_argument('--model', type=str,  choices=[ 'anomalous', 'adone', 'cola', 'conad', 'dominant', 'dmgd', 'done', 'gaan', 'gadnr', 'gae', 'guide', 'ocgnn', 'one', 'radar', 'scan'], required=False, help='Model to use')
    parser.add_argument('--outlier_type', type=str, choices=['structural', 'contextual', 'dice', 'path', 'cont_struc', 'path_dice'] , required=True, help='Model to use')
    args = parser.parse_args()
    args.struc_drop_prob = 0.2
    args.dice_ratio = 0.5
    args.outlier_seed = 0
    
    n = 2000
    z_dim = 50
    p = 0.4
    q = 0.3
    alpha = 0.01  
    beta = 0.01
    threshold = 0.6
    dim = 32
    dataset = 'synthetic'
    path_root = 'synthetic/'
    path = path_root+f'synthetic_{n}_{z_dim}_{p}_{q}_{alpha}_{beta}_{threshold}_{dim}.mat'
    synthetic_data = generate_synthetic_data(path, n, z_dim, p, q, alpha, beta, threshold, dim)
    x, edge_index, labels, sens, x_cf, edge_index_cf, edge_num = synthetic_data["x"], synthetic_data["edge_index"], synthetic_data["labels"], synthetic_data["sens"], synthetic_data["x_cf"], synthetic_data["edge_index_cf"], synthetic_data["edge_num"]
    
    
    args.outlier_num  = math.ceil(n * 0.05)
    args.struc_clique_size = math.ceil(edge_num)
    args.sample_size = args.struc_clique_size
    
    # x = x.astype(np.float32)
    data = Data(x = np.concatenate([sens.reshape(-1,1), x], axis=1), edge_index = edge_index, y = labels)
    data_mod, data_mod.y = outlier_injection(args, data)
    # data_cf = Data(x = x_cf, edge_index = edge_index_cf, y = labels)
    
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, data_mod, sens), n_trials=50)

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))