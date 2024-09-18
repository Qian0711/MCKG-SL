from .rgcn_model import RGCN
from .gat_model import GAT
from dgl import mean_nodes
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import networkx as nx
from torch.nn.utils.weight_norm import weight_norm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from functools import partial

"""
File based off of dgl tutorial on RGCN
Source: https://github.com/dmlc/dgl/tree/master/examples/pytorch/rgcn
"""


class Cross_Attention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=4,
                 qkv_bias=False,
                 drop_ratio=0.,
                 ):
        super(Cross_Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop_ratio)

    def forward(self, x, y):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv2 = self.qkv(y).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv2[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class Decoder(nn.Module):
    def __init__(self,
                 dim,
                 mlp_ratio=4.,
                 norm_layer=nn.LayerNorm,
                 drop_ratio=0.
                 ):
        super(Decoder, self).__init__()
        self.norm1 = norm_layer(dim)
        # self.attn = Attention(dim, drop_ratio=drop_ratio)
        self.cross_attn = Cross_Attention(dim, drop_ratio=drop_ratio)

        self.drop_path = DropPath(drop_ratio) if drop_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)

        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            out_features=dim,
            drop=drop_ratio
        )

    def forward(self, x, y):
        # y = y + self.drop_path(self.attn(self.norm1(y)))
        out = y + self.drop_path(self.cross_attn(x, self.norm1(y)))
        out = out + self.drop_path(self.mlp(self.norm2(y)))
        return out


class Mlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features,
                 out_features,
                 act_layer=nn.GELU,
                 drop=0.):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GraphClassifier(nn.Module):
    def __init__(self, params):  # in_dim, h_dim, rel_emb_dim, out_dim, num_rels, num_bases):params：命名元组，包括输入维度、隐藏层维度、关系嵌入维度、输出维度、关系数量和基的数量
        super().__init__()


        self.params = params
        self.dropout = nn.Dropout(p = params.dropout)
        self.relu = nn.ReLU()
        # self.hid_dim = params.dind
        self.train_rels = params.train_rels
        self.relations = params.num_rels
        self.gnn = RGCN(params)  # in_dim, h_dim, h_dim, num_rels, num_bases)
        self.gat = GAT(params)

        # MLP
        self.mp_layer1 = nn.Linear(self.params.feat_dim, 256)  # 多层感知机模型的第一层线性层
        self.mp_layer2 = nn.Linear(256, self.params.emb_dim)  # 多层感知机模型的第二层线性层
        self.bn1 = nn.BatchNorm1d(256)  # 批标准化层
        embed_dim = 740
        depth_decoder = 4
        self.depth_decoder = depth_decoder
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm_decoder = norm_layer(64)
        self.norm_decoder2 = norm_layer(64)
        self.decoder = Decoder(dim=64, mlp_ratio=4, drop_ratio=0., )
        self.decoder2 = Decoder(dim=64, mlp_ratio=4, drop_ratio=0., )
        self.sig = nn.Sigmoid()

        # Decoder
        #全连接层用于特征融合
        # 判断是否需要添加头实体嵌入（head entity embedding）和尾实体嵌入（tail entity embedding）
        if self.params.add_ht_emb and self.params.add_sb_emb:
            # 判断是否需要添加特征嵌入（feature embedding）和转换嵌入（transE embedding）
            # params.emb_dim=params.emb_dim*2
            if self.params.add_feat_emb and self.params.add_transe_emb:
                # self.fc_layer = nn.Linear(3 * (1 + self.params.num_gcn_layers) * (
                #             self.params.emb_dim + self.params.inp_dim) + 2 * self.params.emb_dim, 512)
                self.fc_layer = nn.Linear(1808, 512)
            elif self.params.add_feat_emb:
                self.fc_layer = nn.Linear(
                    3 * (self.params.num_gcn_layers) * self.params.emb_dim + 2 * self.params.emb_dim, 512)
            else:
                self.fc_layer = nn.Linear(
                    3 * (1 + self.params.num_gcn_layers) * (self.params.emb_dim + self.params.inp_dim), 512)
        elif self.params.add_ht_emb:
            self.fc_layer = nn.Linear(2 * (1 + self.params.num_gcn_layers) * self.params.emb_dim, 512)
        else:
            self.fc_layer = nn.Linear(self.params.num_gcn_layers * self.params.emb_dim, 512)
        # 设置两个额外的全连接层，分别将输入维度512降为128，和128降为1
        self.fc_layer_d = nn.Linear(embed_dim,64)
        self.fc_layer_t = nn.Linear(embed_dim,64)
        # self.fc_layer_2 = nn.Linear(128, 1)

        self.fc_layer_3 = nn.Linear(512, 1)#4*2*64

        print('fc',self.fc_layer)
    def omics_feat(self, emb):
        self.genefeat = emb

    def get_omics_features(self, ids):
        a = []
        for i in ids:
            a.append(self.genefeat[i.cpu().numpy().item()])
        return np.array(a)


    def forward(self, data):
        #该模型以图g为输入，并对其进行多次操作以获得预测。
        g = data
        g.ndata['h'] = self.gnn(g)
        # print(g.ndata['h'])
        g1_out = mean_nodes(g, 'repr')
        #print('g_out.shape=',g1_out.shape)
        g2 = data
        nx_g2 = g2.to_networkx()
        adj_matrix = nx.adjacency_matrix(nx_g2)
        adj_tensor = torch.tensor(adj_matrix.todense(), dtype=torch.float32).cuda()
        g2.ndata['h'] = self.gat(g2.ndata['feat'], adj_tensor)
        g2_out = mean_nodes(g2, 'repr')
        #print('g2_out.shape=', g2_out.shape)
        g_out = torch.cat([g1_out, g2_out], dim=2)
        #从节点数据字典中获取头结点的索引并保存在变量 head_ids 中
        # 获取头结点的表示，并将结果保存在变量 head_embs 中

        head_ids = (g.ndata['id'] == 1).nonzero().squeeze(1)
        head_embs1 = g.ndata['repr'][head_ids]
        head_embs2 = g2.ndata['repr'][head_ids]
        head_embs=torch.cat([head_embs1,head_embs2], dim=2)
        #同上操作tail节点
        tail_ids = (g.ndata['id'] == 2).nonzero().squeeze(1)
        tail_embs1 = g.ndata['repr'][tail_ids]
        tail_embs2 = g2.ndata['repr'][tail_ids]
        tail_embs = torch.cat([tail_embs1, tail_embs2], dim=2)
        #调用 get_omics_features 方法获取头结点和尾节点的基因表达特征向量，将结果保存在 head_feat 和 tail_feat 中，并转换为 pytorch 的 FloatTensor 格式，并将其移到指定的计算设备上（如 GPU）
        head_feat = torch.FloatTensor(self.get_omics_features(g.ndata['idx'][head_ids])).to(self.params.device)
        tail_feat = torch.FloatTensor(self.get_omics_features(g.ndata['idx'][tail_ids])).to(self.params.device)

        head_feat_expanded = head_feat.unsqueeze(1)
        head_feat_expanded = head_feat_expanded.repeat(1, head_embs.size(1), 1)
        tail_feat_expanded = tail_feat.unsqueeze(1)
        tail_feat_expanded = tail_feat_expanded.repeat(1, head_embs.size(1), 1)

        feature1 = torch.cat((head_embs, head_feat_expanded), dim=2)
        # Head_embs{64,4,140}与Head_feat{64,600}拼接成feat1：   {64,4,740}
        feature2 = torch.cat((tail_embs, tail_feat_expanded), dim=2)

        # current_shape = (64, 4, 740)
        feat1 = self.relu(self.fc_layer_d(feature1))
        feat2 = self.relu(self.fc_layer_t(feature2))

        # 目标大小
        # target_shape = (128, 64, 64)

        # current_size = feat1.numel()
        # target_size = torch.prod(torch.tensor(target_shape))

        # 需要添加的零元素数量
        # elements_to_add = target_size - current_size

        # 添加零元素
        # zeros_to_add = torch.zeros(elements_to_add, device='cuda:0')
        #
        # # 将当前张量扁平化，并添加零元素
        # flattened_current_tensor = torch.cat([feat1.view(-1), zeros_to_add])
        #
        # # 重新调整形状为目标大小
        # reshaped_tensor1 = flattened_current_tensor.view(target_shape).clone().detach().requires_grad_(True)
        # # feat1变换为{128,64,64}
        # flattened_current_tensor = torch.cat([feat2.view(-1), zeros_to_add])
        #
        # # 重新调整形状为目标大小
        # reshaped_tensor2 = flattened_current_tensor.view(target_shape).clone().detach().requires_grad_(True)
        # feat2变换为{128,64,64}


        # self.decoder = Decoder(dim=embed_dim, mlp_ratio=4, drop_ratio=0., )
        # self.decoder2 = Decoder(dim=embed_dim, mlp_ratio=4, drop_ratio=0., )
        #
        # reshaped_tensor1 = reshaped_tensor1.to('cpu')
        # reshaped_tensor2 = reshaped_tensor2.to('cpu')
        # x = reshaped_tensor1
        # y = reshaped_tensor2

        # x = self.decoder(feat1)
        # y = self.decoder2(feat2)

        # cross1
        out1 = self.decoder(feat1, feat2)
        for i in range(self.depth_decoder - 1):
            out1 = self.decoder(feat1, out1)
        out1 = self.norm_decoder(out1)
        # out1 = out1.reshape(B, 1, 256, -1)

        # cross2
        out2 = self.decoder2(feat2, feat1)
        for i in range(self.depth_decoder - 1):
            out2 = self.decoder2(feat2, out2)
        out2 = self.norm_decoder2(out2)
        # out2 = out2.reshape(B, 1, 256, -1)

        concatenated_tensor = torch.cat((out1, out2), dim=-1)


        # reshaped_tensor = concatenated_tensor.view(64, 1808)
        # reshaped_tensor = concatenated_tensor.view(64, 16384)
        # (128+128)*64*64=256*84*64==》64*16384
        # 打印调整后的张量大小
        # print(reshaped_tensor.size())  # 应该输出 torch.Size([64, 16384])

        # g_rep = reshaped_tensor.to('cuda')
        # g_rep:(64,16384)
        g_rep = torch.reshape(concatenated_tensor,(concatenated_tensor.size(0),-1))
        # output = self.fc_layer_3(self.relu(self.dropout(g_rep)))
        output = self.sig(self.fc_layer_3((g_rep)))
        # output:(64,1)
        output = output.squeeze(-1)#加了squeeze之后迭代速度增快了


        # output:(64)
        return (output, g_rep)
