import argparse
import json
import os
import torch
import numpy as np
from torch_geometric.data import Data
from scipy.sparse import csc_matrix
from pygod.detector import AdONE, ANOMALOUS, AnomalyDAE, CoLA, CONAD, DMGD, DOMINANT, DONE, GAAN, GADNR, GAE, GUIDE
# from models.custom_dominant import CustomDOMINANT  # Import your custom model
from pygod.metric import eval_roc_auc, eval_average_precision
import optuna
import pickle
import logging

# 模型映射
model_map = {
    'AdONE': AdONE, 'ANOMALOUS': ANOMALOUS, 'AnomalyDAE': AnomalyDAE,
    'CoLA': CoLA, 'CONAD': CONAD, 'DMGD': DMGD, 'DOMINANT': DOMINANT,
    'DONE': DONE, 'GAAN': GAAN, 'GADNR': GADNR, 'GAE': GAE, 'GUIDE': GUIDE
}

def load_data(pkl_file_path):
    with open(pkl_file_path, 'rb') as f:
        data = pickle.load(f)
    
    x = data.x
    edge_index = data.edge_index
    y = data.y
    
    data = Data(x=x, edge_index=edge_index, y=y)
    return data

def objective(trial, data, model_class):
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

    model = model_class(contamination=0.05, gpu=0, **model_params)
    model.fit(data)
    # anomaly_scores = model.decision_function(data)
    y_score = np.array(model.decision_score_).ravel()
    threshold_percentile=95
    threshold = np.percentile(y_score, threshold_percentile)
    y_pred = (y_score > threshold).astype(int)

    roc_auc = eval_roc_auc(data.y, y_pred)
    average_precision = eval_average_precision(data.y, y_pred)
    
    trial.set_user_attr("roc_auc", roc_auc)
    trial.set_user_attr("average_precision", average_precision)
    trial.set_user_attr("params", model_params)

    logging.info(f"Trial {trial.number} - ROC AUC: {roc_auc:.6f}, Average Precision: {average_precision:.6f}")
    logging.info(f"Parameters: {json.dumps(model_params, indent=4)}")

    return roc_auc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='PKL file path')
    parser.add_argument('--model', type=str, required=True, help='Model to use')
    parser.add_argument('--n_trials', type=int, default=20, help='Number of Optuna trials')
    parser.add_argument('--output_dir', type=str, default='results', help='Directory to save results')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    data = load_data(args.dataset)
    model_class = model_map[args.model]

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, data, model_class), n_trials=args.n_trials)

    best_trial = study.best_trial
    

    results = {
        "value": best_trial.value,
        "params": best_trial.params,
        "roc_auc": best_trial.user_attrs["roc_auc"],
        "average_precision": best_trial.user_attrs["average_precision"]
    }

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f"{args.model}_results.json")

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

    logging.info("Best trial:")
    logging.info(f"  Value: {best_trial.value}")
    logging.info("  Params: ")
    for key, value in best_trial.params.items():
        logging.info(f"    {key}: {value}")

    logging.info(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
