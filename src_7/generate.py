import sys
import numpy as np
import torch
import random
import argparse
import math
import os

from utils.Preprocessing import load_data
from utils.loader_HNN_GAD import *

from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix
import pickle



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--flip_rate', type=float, default=0.0)
    parser.add_argument('--dataset', type=str, default='bail', help='Dataset to use')
    args = parser.parse_args()
    
    adj, features, labels, idx_train, idx_val, idx_test, sens, sens_idx, raw_data_info = load_data(path_root='', dataset=args.dataset)
    label_number = len(labels) 
    
    # Ensure all data is in torch.Tensor format
    features = torch.tensor(features, dtype=torch.float)
    labels = torch.tensor(labels, dtype=torch.long)
    sens = torch.tensor(sens, dtype=torch.float)

    edge_index, edge_attr = from_scipy_sparse_matrix(adj)
    data = Data(x=features, edge_index=edge_index, y=labels)

    args.struc_drop_prob = 0.2
    args.dice_ratio = 0.5
    args.outlier_seed = 0
    args.struc_clique_size = math.ceil(np.mean(np.sum(adj, axis=1)))
    args.sample_size = args.struc_clique_size
    args.outlier_num = math.ceil(data.num_nodes * 0.05)
    
    outlier_types = ['structural', 'contextual', 'dice', 'path', 'cont_struc', 'path_dice']
    for outlier_type in outlier_types:
        args.outlier_type = outlier_type
        data_mod, y_outlier = outlier_injection(args, data)
        data_mod.y = y_outlier

        filename = f"injected_data/{args.dataset}_{outlier_type}.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(data_mod, f)
        print(f"Data with {outlier_type} outliers has been generated and saved as {filename}.")

