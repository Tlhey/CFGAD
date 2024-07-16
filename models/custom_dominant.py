import torch.nn as nn
import torch.nn.functional as F
import torch
from .layers import GraphConvolution

class Encoder(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(Encoder, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2_s = GraphConvolution(nhid, nhid)
        self.gc2_ns = GraphConvolution(nhid, nhid)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        z_s = F.relu(self.gc2_s(x, adj))
        z_ns = F.relu(self.gc2_ns(x, adj))
        return z_s, z_ns

class Attribute_Decoder(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(Attribute_Decoder, self).__init__()

        self.gc1 = GraphConvolution(nhid, nhid)
        self.gc2 = GraphConvolution(nhid, nfeat)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        return x

class Structure_Decoder(nn.Module):
    def __init__(self, nhid, dropout):
        super(Structure_Decoder, self).__init__()

        self.gc1 = GraphConvolution(nhid, nhid)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = x @ x.T
        return x

class CustomDOMINANT(nn.Module):
    def __init__(self, feat_size, hidden_size, dropout):
        super(CustomDOMINANT, self).__init__()
        
        self.shared_encoder = Encoder(feat_size, hidden_size, dropout)
        self.attr_decoder_s = Attribute_Decoder(feat_size, hidden_size, dropout)
        self.attr_decoder_ns = Attribute_Decoder(feat_size, hidden_size, dropout)
        self.struct_decoder = Structure_Decoder(hidden_size, dropout)
    
    def forward(self, x, adj):
        # encode
        z_s, z_ns = self.shared_encoder(x, adj)
        # decode feature matrix
        x_s_hat = self.attr_decoder_s(z_s, adj)
        x_ns_hat = self.attr_decoder_ns(z_ns, adj)
        # decode adjacency matrix
        struct_reconstructed = self.struct_decoder(z_ns, adj)
        # return reconstructed matrices
        return struct_reconstructed, x_s_hat, x_ns_hat, z_s, z_ns

    def custom_loss_function(self, reconstructions, original, z_s, z_ns):
        x, adj = original
        struct_reconstructed, x_s_hat, x_ns_hat = reconstructions

        # Reconstruction loss
        loss_x = F.mse_loss(x, x_s_hat + x_ns_hat)
        loss_a = F.mse_loss(adj, struct_reconstructed)

        # Counterfactual fairness loss
        z_s_cf = self.flip_sensitive_attributes(z_s)
        x_cf_rec = self.attr_decoder_s(z_s_cf, adj)
        loss_cf = F.mse_loss(x_s_hat, x_cf_rec)

        # Disentanglement loss using Total Correlation
        tc_loss = self.total_correlation(z_s, z_ns)

        # Combine all losses
        loss = loss_x + loss_a + loss_cf + tc_loss
        return loss

    def flip_sensitive_attributes(self, z_s):
        # Implement your method to flip sensitive attributes here
        return 1 - z_s

    def mutual_info(self, z_s, z_ns):
        # Implement your mutual information calculation here
        pass

    def total_correlation(self, z_s, z_ns):
        """
        Compute the Total Correlation Loss to ensure disentanglement
        between z_s and z_ns.
        """
        joint_distribution = torch.cat([z_s, z_ns], dim=-1)
        marginal_distribution_zs = z_s
        marginal_distribution_zns = z_ns
        
        tc_loss = (torch.mean(torch.log(joint_distribution)) -
                   torch.mean(torch.log(marginal_distribution_zs)) -
                   torch.mean(torch.log(marginal_distribution_zns)))
        return tc_loss

    def fit(self, data, *args, **kwargs):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        for epoch in range(100):
            optimizer.zero_grad()
            z_s, z_ns = self.shared_encoder(data.x, data.edge_index)
            x_s_hat = self.attr_decoder_s(z_s, data.edge_index)
            x_ns_hat = self.attr_decoder_ns(z_ns, data.edge_index)
            struct_reconstructed = self.struct_decoder(z_ns, data.edge_index)
            reconstructions = (struct_reconstructed, x_s_hat, x_ns_hat)
            original = (data.x, data.edge_index)
            loss = self.custom_loss_function(reconstructions, original, z_s, z_ns)
            loss.backward()
            optimizer.step()
