import os
import math
import numpy as np
import argparse
import scipy.io as scio
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix
import pickle
import sys

# 添加项目根目录到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.loader_HNN_GAD import outlier_injection

def load_synthetic(path):
    data = scio.loadmat(path)
    features = data['x']
    features_cf = data['x_cf']
    adj = data['adj']
    adj_cf = data['adj_cf']
    labels = data['y']
    labels_cf = data['y_cf']
    sens = data['sens'][0]
    sens_cf = data['sens_cf'][0]
    features = np.concatenate([sens.reshape(-1, 1), features], axis=1)
    features_cf = np.concatenate([sens_cf.reshape(-1, 1), features_cf], axis=1)
    raw_data_info = {
        'adj': adj,
        'w': data['w'],
        'w_s': data['w_s'][0][0],
        'z': data['z'],
        'v': data['v'],
        'feat_idxs': data['feat_idxs'][0],
        'alpha': data['alpha'][0][0]
    }
    features = torch.FloatTensor(features)
    features_cf = torch.FloatTensor(features_cf)
    labels = torch.LongTensor(labels).squeeze()
    labels_cf = torch.LongTensor(labels_cf).squeeze()
    sens = torch.FloatTensor(sens)
    sens_cf = torch.FloatTensor(sens_cf)



    return sens, adj, features, labels, sens_cf, adj_cf, features_cf, labels_cf, raw_data_info

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to synthetic data MAT file')
    args = parser.parse_args()

    # 加载合成数据
    sens, adj, features, labels, sens_cf, adj_cf, features_cf, labels_cf, raw_data_info = load_synthetic(args.data_path)

    # Ensure all data is in torch.Tensor format
    features = torch.tensor(features, dtype=torch.float)
    labels = torch.tensor(labels, dtype=torch.long)
    sens = torch.tensor(sens, dtype=torch.float)

    edge_index, edge_attr = from_scipy_sparse_matrix(adj)
    edge_index = torch.tensor(edge_index, dtype=torch.long)  # 确保 edge_index 是一个 PyTorch 张量
    data = Data(x=features, edge_index=edge_index, y=labels)

    args.struc_drop_prob = 0.2
    args.dice_ratio = 0.5
    args.outlier_seed = 0
    args.struc_clique_size = math.ceil(np.mean(np.sum(adj, axis=1)))
    args.sample_size = args.struc_clique_size
    args.outlier_num = math.ceil(data.num_nodes * 0.05)

    outlier_types = ['structural', 'contextual', 'dice', 'path', 'cont_struc', 'path_dice']
    
    if not os.path.exists('data/processed'):
        os.makedirs('data/processed')

    for outlier_type in outlier_types:
        args.outlier_type = outlier_type
        data_mod, y_outlier = outlier_injection(args, data)
        data_mod.y = y_outlier

        filename = f"data/processed/synthetic_{outlier_type}.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(data_mod, f)
        print(f"Data with {outlier_type} outliers has been generated and saved as {filename}.")
