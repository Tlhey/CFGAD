import sys
import os
import numpy as np
import torch
import random
import argparse
import math
import pickle

from pygod.detector import AdONE, ANOMALOUS, AnomalyDAE, CoLA, CONAD, DMGD, DOMINANT, DONE, GAAN, GADNR, GAE, GUIDE
from torch_geometric.data import Data
from pygod.detector import DOMINANT
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import optuna

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


def objective(trial, data_list, sens, model_class):
    # Hyperparameter suggestions based on model type
    common_params = {
        'lr': trial.suggest_float("lr", 1e-5, 1e-1, log=True),
        'dropout': trial.suggest_float("dropout", 1e-6, 1, log=True),
        'weight_decay': trial.suggest_float("weight_decay", 1e-6, 1, log=True),
        'num_layers': trial.suggest_categorical("num_layers", [2, 3, 4, 5]),
        'hid_dim': trial.suggest_categorical("hid_dim", [32, 64, 128, 256])
    }

    if model_class in [AdONE, CONAD, DMGD, DOMINANT, GUIDE]:
        model_params = common_params
    elif model_class == ANOMALOUS:
        model_params = {**common_params, 'gamma': trial.suggest_float("gamma", 1e-2, 1e2, log=True)}
    elif model_class == AnomalyDAE:
        model_params = {**common_params, 'emb_dim': trial.suggest_categorical("emb_dim", [32, 64, 128, 256])}
    elif model_class == DONE:
        model_params = {**common_params, 'w1': trial.suggest_float("w1", 1e-5, 1, log=True),
                        'w2': trial.suggest_float("w2", 1e-5, 1, log=True),
                        'w3': trial.suggest_float("w3", 1e-5, 1, log=True),
                        'w4': trial.suggest_float("w4", 1e-5, 1, log=True),
                        'w5': trial.suggest_float("w5", 1e-5, 1, log=True)}
    elif model_class == GAAN:
        model_params = {**common_params, 'noise_dim': trial.suggest_categorical("noise_dim", [4, 8, 16, 32, 64])}
    elif model_class == GADNR:
        model_params = {**common_params, 'sample_size': trial.suggest_int("sample_size", 1, 10),
                        'sample_time': trial.suggest_int("sample_time", 1, 10),
                        'lambda_loss1': trial.suggest_float("lambda_loss1", 1e-5, 1, log=True),
                        'lambda_loss2': trial.suggest_float("lambda_loss2", 1e-5, 1, log=True),
                        'lambda_loss3': trial.suggest_float("lambda_loss3", 1e-5, 1, log=True)}
    else:
        model_params = common_params

    def train_and_predict(data, contamination=0.05, threshold_percentile=95):
        model = model_class(contamination=contamination, gpu=0, **model_params)
        model.fit(data)
        y_score = np.array(model.decision_score_).ravel()
        threshold = np.percentile(y_score, threshold_percentile)
        y_pred = (y_score > threshold).astype(int)
        return y_score, y_pred, threshold

    outlier_types = ['structural', 'contextual', 'dice', 'path', 'cont_struc', 'path_dice']
    eval_result, auc = {}, 0
    total_auc = 0

    for outlier_type in outlier_types:
        data = data_list[outlier_type]
        y_score, y_pred, threshold = train_and_predict(data)
        eval_result[outlier_type] = evaluate_outlier_detection(data.y, y_pred, sens)
        print(f"Trial {trial.number} {outlier_type}:", eval_result[outlier_type])
        total_auc += eval_result[outlier_type]['auc']

    average_auc = total_auc / len(outlier_types)
    return average_auc


def load_data_mod(dataset, outlier_type):
    filename = f"injected_data/{dataset}_{outlier_type}.pkl"
    with open(filename, 'rb') as f:
        data_mod = pickle.load(f)
    return data_mod

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='bail', help='Dataset to use')
    parser.add_argument('--model', type=str, required=True, help='Model to use')
    args = parser.parse_args()
    
    model_map = { 'AdONE': AdONE, 'ANOMALOUS': ANOMALOUS, 'AnomalyDAE': AnomalyDAE, 'CoLA': CoLA, 'CONAD': CONAD, 'DMGD': DMGD, 'DOMINANT': DOMINANT, 'DONE': DONE, 'GAAN': GAAN, 'GADNR': GADNR, 'GAE': GAE, 'GUIDE': GUIDE}

    model_class = model_map[args.model]

    outlier_types = ['structural', 'contextual', 'dice', 'path', 'cont_struc', 'path_dice']
    data_mod_list = {}
    
    for outlier_type in outlier_types:
        data_mod_list[outlier_type] = load_data_mod(args.dataset, outlier_type)
        print(f"Data with {outlier_type} outliers has been loaded.")

    sens = data_mod_list[outlier_types[0]].x[:, 0] # the sensitive attribute is the first column

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, data_mod_list, sens, model_class), n_trials=20)

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
