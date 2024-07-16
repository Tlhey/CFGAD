# import argparse
# import json
# import os
# import torch
# import numpy as np
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.data import Data
# from scipy.sparse import csc_matrix
# from pygod.detector import AdONE, ANOMALOUS, AnomalyDAE, CoLA, CONAD, DMGD, DONE, GAAN, GADNR, GAE, GUIDE
# from pygod.metric import eval_roc_auc, eval_average_precision
# import optuna
# import pickle
# import logging
# from models.custom_dominant import CustomDOMINANT  # Import your custom model
# from models.layers import GraphConvolution
# from torch_geometric.utils import to_scipy_sparse_matrix

# def get_device():
#     return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# class Encoder(nn.Module):
#     def __init__(self, nfeat, nhid, dropout):
#         super(Encoder, self).__init__()

#         self.gc1 = GraphConvolution(nfeat, nhid)
#         self.gc2_s = GraphConvolution(nhid, nhid)
#         self.gc2_ns = GraphConvolution(nhid, nhid)
#         self.dropout = dropout

#     def forward(self, x, adj):
#         x = F.relu(self.gc1(x, adj))
#         x = F.dropout(x, self.dropout, training=self.training)
#         z_s = F.relu(self.gc2_s(x, adj))
#         z_ns = F.relu(self.gc2_ns(x, adj))
#         return z_s, z_ns

# class Attribute_Decoder(nn.Module):
#     def __init__(self, nfeat, nhid, dropout):
#         super(Attribute_Decoder, self).__init__()

#         self.gc1 = GraphConvolution(nhid, nhid)
#         self.gc2 = GraphConvolution(nhid, nfeat)
#         self.dropout = dropout

#     def forward(self, x, adj):
#         x = F.relu(self.gc1(x, adj))
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = F.relu(self.gc2(x, adj))
#         return x

# class Structure_Decoder(nn.Module):
#     def __init__(self, nhid, dropout):
#         super(Structure_Decoder, self).__init__()

#         self.gc1 = GraphConvolution(nhid, nhid)
#         self.dropout = dropout

#     def forward(self, x, adj):
#         x = F.relu(self.gc1(x, adj))
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = x @ x.T
#         return x

# class CustomDOMINANT(nn.Module):
#     def __init__(self, feat_size, hidden_size, dropout):
#         super(CustomDOMINANT, self).__init__()
#         self.device = get_device()
        
#         self.shared_encoder = Encoder(feat_size, hidden_size, dropout)
#         self.attr_decoder_s = Attribute_Decoder(feat_size, hidden_size, dropout)
#         self.attr_decoder_ns = Attribute_Decoder(feat_size, hidden_size, dropout)
#         self.struct_decoder = Structure_Decoder(hidden_size, dropout)
    
#     def forward(self, x, adj):
#         # encode
#         z_s, z_ns = self.shared_encoder(x, adj)
#         # decode feature matrix
#         x_s_hat = self.attr_decoder_s(z_s, adj)
#         x_ns_hat = self.attr_decoder_ns(z_ns, adj)
#         # decode adjacency matrix
#         struct_reconstructed = self.struct_decoder(z_ns, adj)
#         # return reconstructed matrices
#         return struct_reconstructed.to(self.device), x_s_hat.to(self.device), x_ns_hat.to(self.device), z_s.to(self.device), z_ns.to(self.device)

#     def custom_loss_function(self, reconstructions, original, z_s, z_ns):
#         x, adj = original
#         struct_reconstructed, x_s_hat, x_ns_hat = reconstructions

#         # Reconstruction loss
#         loss_x = F.mse_loss(x, x_s_hat + x_ns_hat)
#         loss_a = F.mse_loss(adj, struct_reconstructed)

#         # Counterfactual fairness loss
#         z_s_cf = self.flip_sensitive_attributes(z_s)
#         x_cf_rec = self.attr_decoder_s(z_s_cf, adj)
#         loss_cf = F.mse_loss(x_s_hat, x_cf_rec)

#         # Disentanglement loss using Total Correlation
#         tc_loss = self.total_correlation(z_s, z_ns)

#         # Combine all losses
#         loss = loss_x + loss_a + loss_cf + tc_loss
#         return loss.to(self.device)

#     def flip_sensitive_attributes(self, z_s):
#         # Implement your method to flip sensitive attributes here
#         return (1 - z_s).to(self.device)

#     def mutual_info(self, z_s, z_ns):
#         # Implement your mutual information calculation here
#         pass

#     def total_correlation(self, z_s, z_ns):
#         """
#         Compute the Total Correlation Loss to ensure disentanglement
#         between z_s and z_ns.
#         """
#         joint_distribution = torch.cat([z_s, z_ns], dim=-1)
#         marginal_distribution_zs = z_s
#         marginal_distribution_zns = z_ns
        
#         tc_loss = (torch.mean(torch.log(joint_distribution)) -
#                    torch.mean(torch.log(marginal_distribution_zs)) -
#                    torch.mean(torch.log(marginal_distribution_zns)))
#         return tc_loss.to(self.device)
    
#     #anomaly score part
#     def anomaly_score(self, adj, attrs):
#         A_hat, X_hat, _, z_s, z_ns = self(attrs, adj)
        
#         # Attribute reconstruction error
#         diff_attribute = torch.pow(X_hat - attrs, 2)
#         attribute_reconstruction_errors = torch.sqrt(torch.sum(diff_attribute, 1))
        
#         # Structure reconstruction error
#         diff_structure = torch.pow(A_hat - adj, 2)
#         structure_reconstruction_errors = torch.sqrt(torch.sum(diff_structure, 1))
        
#         # Combine errors (you can adjust the alpha value)
#         alpha = 0.8
#         anomaly_scores = alpha * attribute_reconstruction_errors + (1 - alpha) * structure_reconstruction_errors
        
#         return anomaly_scores

#     def decision_function(self, data):
#         self.eval()
#         with torch.no_grad():
#             return self.anomaly_score(data.adj, data.x).cpu().numpy()


#     def fit(self, data, *args, **kwargs):
#         optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
#         for epoch in range(100):
#             optimizer.zero_grad()
#             z_s, z_ns = self.shared_encoder(data.x, data.adj)
#             x_s_hat = self.attr_decoder_s(z_s, data.adj)
#             x_ns_hat = self.attr_decoder_ns(z_ns, data.adj)
#             struct_reconstructed = self.struct_decoder(z_ns, data.adj)
#             reconstructions = (struct_reconstructed, x_s_hat, x_ns_hat)
#             original = (data.x, data.adj)
#             loss = self.custom_loss_function(reconstructions, original, z_s, z_ns)
#             loss.backward()
#             optimizer.step()

# def load_data(pkl_file_path):
#     with open(pkl_file_path, 'rb') as f:
#         data = pickle.load(f)
    
#     x = data.x
#     edge_index = data.edge_index
#     y = data.y

#     # Convert edge_index to sparse adjacency matrix
#     device = get_device()
#     adj = to_scipy_sparse_matrix(edge_index, num_nodes=x.size(0))
#     adj = torch.FloatTensor(adj.todense()).to(device)
    
#     x = torch.FloatTensor(data.x).to(device)
#     edge_index = torch.LongTensor(data.edge_index).to(device)
#     y = torch.LongTensor(data.y).to(device)

#     data = Data(x=x, edge_index=edge_index, y=y, adj = adj)
#     return data

# def objective(trial, data):
#     params = {
#         'lr': trial.suggest_float("lr", 1e-5, 1e-1, log=True),
#         'dropout': trial.suggest_float("dropout", 1e-6, 1, log=True),
#         'weight_decay': trial.suggest_float("weight_decay", 1e-6, 1, log=True),
#         'num_layers': trial.suggest_categorical("num_layers", [2, 3, 4, 5]),
#         'hid_dim': trial.suggest_categorical("hid_dim", [32, 64, 128, 256])
#     }

#     device = get_device()
#     model = CustomDOMINANT(feat_size=data.num_features, hidden_size=params['hid_dim'], dropout=params['dropout']).to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])

#     model.train()
#     for epoch in range(100):  # Adjust epoch number as needed
#         optimizer.zero_grad()
#         z_s, z_ns = model.shared_encoder(data.x, data.adj)
#         x_s_hat = model.attr_decoder_s(z_s, data.adj)
#         x_ns_hat = model.attr_decoder_ns(z_ns, data.adj)
#         struct_reconstructed = model.struct_decoder(z_ns, data.adj)
#         reconstructions = (struct_reconstructed, x_s_hat, x_ns_hat)
#         original = (data.x, data.adj)
#         loss = model.custom_loss_function(reconstructions, original, z_s, z_ns)
#         loss.backward()
#         optimizer.step()

#     model.eval()
#     with torch.no_grad():
#         # struct_reconstructed, x_s_hat, x_ns_hat, z_s, z_ns = model(data.x, data.adj)
#         y_score = model.decision_function(data)
#         threshold_percentile = 95
#         threshold = np.percentile(y_score, threshold_percentile)
#         # y_pred = (y_score > threshold).astype(int)
#         y_pred = (y_score > threshold).cpu().numpy().astype(int)
        
#     roc_auc = eval_roc_auc(data.y, y_pred)
#     average_precision = eval_average_precision(data.y, y_pred)
    
#     trial.set_user_attr("roc_auc", roc_auc)
#     trial.set_user_attr("average_precision", average_precision)
#     trial.set_user_attr("params", params)

#     logging.info(f"Trial {trial.number} - ROC AUC: {roc_auc:.6f}, Average Precision: {average_precision:.6f}")
#     logging.info(f"Parameters: {json.dumps(params, indent=4)}")

#     return roc_auc

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--dataset', type=str, required=True, help='PKL file path')
#     parser.add_argument('--n_trials', type=int, default=20, help='Number of Optuna trials')
#     parser.add_argument('--output_dir', type=str, default='results', help='Directory to save results')
#     args = parser.parse_args()

#     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#     device = get_device()
#     logging.info(f"Using device: {device}")

#     data = load_data(args.dataset)

#     study = optuna.create_study(direction="maximize")
#     study.optimize(lambda trial: objective(trial, data), n_trials=args.n_trials)

#     best_trial = study.best_trial

#     results = {
#         "value": best_trial.value,
#         "params": best_trial.params,
#         "roc_auc": best_trial.user_attrs["roc_auc"],
#         "average_precision": best_trial.user_attrs["average_precision"]
#     }

#     os.makedirs(args.output_dir, exist_ok=True)
#     output_file = os.path.join(args.output_dir, f"dominant_results.json")

#     with open(output_file, 'w') as f:
#         json.dump(results, f, indent=4)

#     logging.info("Best trial:")
#     logging.info(f"  Value: {best_trial.value}")
#     logging.info("  Params: ")
#     for key, value in best_trial.params.items():
#         logging.info(f"    {key}: {value}")

#     logging.info(f"Results saved to {output_file}")

# if __name__ == "__main__":
#     main()
import argparse
import json
import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from scipy.sparse import csc_matrix
from pygod.metric import eval_roc_auc, eval_average_precision
import optuna
import pickle
import logging
from torch_geometric.utils import to_scipy_sparse_matrix

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight.data)
        if self.bias is not None:
            nn.init.zeros_(self.bias.data)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class Encoder(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(Encoder, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x

class Attribute_Decoder(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(Attribute_Decoder, self).__init__()
        self.gc1 = GraphConvolution(nhid, nhid)
        self.gc2 = GraphConvolution(nhid, nfeat)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x

class Structure_Decoder(nn.Module):
    def __init__(self, nhid, dropout):
        super(Structure_Decoder, self).__init__()
        self.gc = GraphConvolution(nhid, nhid)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = x @ x.t()
        return torch.sigmoid(x)

class CustomDOMINANT(nn.Module):
    def __init__(self, feat_size, hidden_size, dropout):
        super(CustomDOMINANT, self).__init__()
        self.device = get_device()
        self.encoder = Encoder(feat_size, hidden_size, dropout)
        self.attr_decoder = Attribute_Decoder(feat_size, hidden_size, dropout)
        self.struct_decoder = Structure_Decoder(hidden_size, dropout)
    
    def forward(self, x, adj):
        # encode
        z = self.encoder(x, adj)
        # decode feature matrix
        X_hat = self.attr_decoder(z, adj)
        # decode adjacency matrix
        A_hat = self.struct_decoder(z, adj)
        return A_hat, X_hat, z

    def anomaly_score(self, adj, attrs):
        A_hat, X_hat, _ = self(attrs, adj)
        
        # Attribute reconstruction error
        diff_attribute = torch.pow(X_hat - attrs, 2)
        attribute_reconstruction_errors = torch.sqrt(torch.sum(diff_attribute, 1))
        
        # Structure reconstruction error
        diff_structure = torch.pow(A_hat - adj, 2)
        structure_reconstruction_errors = torch.sqrt(torch.sum(diff_structure, 1))
        
        # Combine errors (you can adjust the alpha value)
        alpha = 0.8
        anomaly_scores = alpha * attribute_reconstruction_errors + (1 - alpha) * structure_reconstruction_errors
        
        return anomaly_scores

    def decision_function(self, data):
        self.eval()
        with torch.no_grad():
            return self.anomaly_score(data.adj, data.x).cpu().numpy()

def load_data(pkl_file_path):
    with open(pkl_file_path, 'rb') as f:
        data = pickle.load(f)
    
    device = get_device()
    x = torch.FloatTensor(data.x).to(device)
    edge_index = torch.LongTensor(data.edge_index).to(device)
    y = torch.LongTensor(data.y).to(device)

    # Convert edge_index to sparse adjacency matrix
    adj = to_scipy_sparse_matrix(edge_index, num_nodes=x.size(0))
    adj = torch.FloatTensor(adj.todense()).to(device)
    
    data = Data(x=x, edge_index=edge_index, y=y, adj=adj)
    return data

def objective(trial, data):
    params = {
        'lr': trial.suggest_float("lr", 1e-5, 1e-1, log=True),
        'dropout': trial.suggest_float("dropout", 1e-6, 1, log=True),
        'weight_decay': trial.suggest_float("weight_decay", 1e-6, 1, log=True),
        'hidden_dim': trial.suggest_categorical("hidden_dim", [32, 64, 128, 256]),
        'alpha': trial.suggest_float("alpha", 0, 1)
    }

    device = get_device()
    model = CustomDOMINANT(feat_size=data.num_features, hidden_size=params['hidden_dim'], dropout=params['dropout']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])

    model.train()
    for epoch in range(100):  # Adjust epoch number as needed
        optimizer.zero_grad()
        A_hat, X_hat, _ = model(data.x, data.adj)
        
        # Calculate loss similar to DOMINANT
        diff_attribute = torch.pow(X_hat - data.x, 2)
        attribute_reconstruction_errors = torch.sqrt(torch.sum(diff_attribute, 1))
        attribute_cost = torch.mean(attribute_reconstruction_errors)
        
        diff_structure = torch.pow(A_hat - data.adj, 2)
        structure_reconstruction_errors = torch.sqrt(torch.sum(diff_structure, 1))
        structure_cost = torch.mean(structure_reconstruction_errors)
        
        loss = params['alpha'] * attribute_cost + (1 - params['alpha']) * structure_cost
        
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    model.eval()
    with torch.no_grad():
        y_score = model.decision_function(data)
        threshold_percentile = 95
        threshold = np.percentile(y_score, threshold_percentile)
        y_pred = (y_score > threshold).astype(int)
        
    roc_auc = eval_roc_auc(data.y.cpu().numpy(), y_pred)
    average_precision = eval_average_precision(data.y.cpu().numpy(), y_pred)
    
    trial.set_user_attr("roc_auc", roc_auc)
    trial.set_user_attr("average_precision", average_precision)
    trial.set_user_attr("params", params)

    logging.info(f"Trial {trial.number} - ROC AUC: {roc_auc:.6f}, Average Precision: {average_precision:.6f}")
    logging.info(f"Parameters: {json.dumps(params, indent=4)}")

    return roc_auc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='PKL file path')
    parser.add_argument('--n_trials', type=int, default=20, help='Number of Optuna trials')
    parser.add_argument('--output_dir', type=str, default='results', help='Directory to save results')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    device = get_device()
    logging.info(f"Using device: {device}")

    data = load_data(args.dataset)

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, data), n_trials=args.n_trials)

    best_trial = study.best_trial

    results = {
        "value": best_trial.value,
        "params": best_trial.params,
        "roc_auc": best_trial.user_attrs["roc_auc"],
        "average_precision": best_trial.user_attrs["average_precision"]
    }

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f"dominant_results.json")

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