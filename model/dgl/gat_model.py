import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .aggregators import SumAggregator, MLPAggregator, GRUAggregator

class GAT(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, params, concat=True):

        super(GAT, self).__init__()
        self.max_label_value = params.max_label_value
        self.inp_dim = params.inp_dim
        self.emb_dim = params.emb_dim
        self.out_features = params.emb_dim
        self.attn_rel_emb_dim = params.attn_rel_emb_dim
        self.num_rels = params.num_rels
        self.aug_num_rels = params.aug_num_rels
        self.num_bases = params.num_bases
        self.num_hidden_layers = params.num_gcn_layers
        self.dropout = params.dropout
        self.edge_dropout = params.edge_dropout
        self.has_attn = params.has_attn
        self.num_nodes = params.num_nodes
        self.device = params.device
        self.add_transe_emb = params.add_transe_emb
        self.concat = concat
        self.alpha = 0.01  # 瞎设置的

        if self.has_attn:
            self.attn_rel_emb = nn.Embedding(self.aug_num_rels, self.attn_rel_emb_dim, sparse=False)
        else:
            self.attn_rel_emb = None

        self.W = nn.Parameter(torch.zeros(size=(params.inp_dim, params.emb_dim)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * params.emb_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

        # Define three GAT layers
        self.gat_layer_1 = GATLayer(params, concat)
        self.gat_layer_2 = GATLayer(params, concat)
        self.gat_layer_3 = GATLayer(params, concat)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)

        # First GAT layer
        h = self.gat_layer_1(h, adj)

        # Second GAT layer
        h = self.gat_layer_2(h, adj)

        # Third GAT layer
        h = self.gat_layer_3(h, adj)

        return h


class GATLayer(nn.Module):
    def __init__(self, params, concat=True):
        super(GATLayer, self).__init__()

        self.in_features = params.emb_dim
        self.out_features = params.emb_dim
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(self.in_features, self.out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * self.out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(0.01)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)

        N = h.size()[0]
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15 * torch.ones_like(e)
        zero_vec = zero_vec.cuda()
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, 0.6, training=self.training)

        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime
